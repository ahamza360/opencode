import { Provider } from "@/provider"
import { Log } from "@/util"
import { Context, Effect, Layer, Record } from "effect"
import * as Stream from "effect/Stream"
import { streamText, wrapLanguageModel, type ModelMessage, type Tool, tool, jsonSchema, InvalidToolInputError, NoSuchToolError } from "ai"
import { mergeDeep, pipe } from "remeda"
import { GitLabWorkflowLanguageModel } from "gitlab-ai-provider"
import { ProviderTransform } from "@/provider"
import { Config } from "@/config"
import { Instance } from "@/project/instance"
import type { Agent } from "@/agent/agent"
import type { MessageV2 } from "./message-v2"
import { Plugin } from "@/plugin"
import { SystemPrompt } from "./system"
import { Flag } from "@/flag/flag"
import { Permission } from "@/permission"
import { PermissionID } from "@/permission/schema"
import { Bus } from "@/bus"
import { Wildcard } from "@/util"
import { SessionID } from "@/session/schema"
import { Auth } from "@/auth"
import { Installation } from "@/installation"
import { InstallationVersion } from "@/installation/version"
import { EffectBridge } from "@/effect"
import * as Option from "effect/Option"
import * as OtelTracer from "@effect/opentelemetry/Tracer"

const log = Log.create({ service: "llm" })
export const OUTPUT_TOKEN_MAX = ProviderTransform.OUTPUT_TOKEN_MAX
type Result = Awaited<ReturnType<typeof streamText>>

export type StreamInput = {
  user: MessageV2.User
  sessionID: string
  parentSessionID?: string
  model: Provider.Model
  agent: Agent.Info
  permission?: Permission.Ruleset
  system: string[]
  messages: ModelMessage[]
  small?: boolean
  tools: Record<string, Tool>
  retries?: number
  toolChoice?: "auto" | "required" | "none"
}

export type StreamRequest = StreamInput & {
  abort: AbortSignal
}

export type Event = Result["fullStream"] extends AsyncIterable<infer T> ? T : never

export interface Interface {
  readonly stream: (input: StreamInput) => Stream.Stream<Event, unknown>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/LLM") {}

const live: Layer.Layer<
  Service,
  never,
  Auth.Service | Config.Service | Provider.Service | Plugin.Service | Permission.Service
> = Layer.effect(
  Service,
  Effect.gen(function* () {
    const auth = yield* Auth.Service
    const config = yield* Config.Service
    const provider = yield* Provider.Service
    const plugin = yield* Plugin.Service
    const perm = yield* Permission.Service

    const run = Effect.fn("LLM.run")(function* (input: StreamRequest) {
      const l = log
        .clone()
        .tag("providerID", input.model.providerID)
        .tag("modelID", input.model.id)
        .tag("session.id", input.sessionID)
        .tag("small", (input.small ?? false).toString())
        .tag("agent", input.agent.name)
        .tag("mode", input.agent.mode)
      l.info("stream", {
        modelID: input.model.id,
        providerID: input.model.providerID,
      })

      const [language, cfg, item, info] = yield* Effect.all(
        [
          provider.getLanguage(input.model),
          config.get(),
          provider.getProvider(input.model.providerID),
          auth.get(input.model.providerID),
        ],
        { concurrency: "unbounded" },
      )

      // TODO: move this to a proper hook
      const isOpenaiOauth = item.id === "openai" && info?.type === "oauth"

      const system: string[] = []
      system.push(
        [
          // use agent prompt otherwise provider prompt
          ...(input.agent.prompt ? [input.agent.prompt] : SystemPrompt.provider(input.model)),
          // any custom prompt passed into this call
          ...input.system,
          // any custom prompt from last user message
          ...(input.user.system ? [input.user.system] : []),
        ]
          .filter((x) => x)
          .join("\n"),
      )

      const header = system[0]
      yield* plugin.trigger(
        "experimental.chat.system.transform",
        { sessionID: input.sessionID, model: input.model },
        { system },
      )
      // rejoin to maintain 2-part structure for caching if header unchanged
      if (system.length > 2 && system[0] === header) {
        const rest = system.slice(1)
        system.length = 0
        system.push(header, rest.join("\n"))
      }

      const variant =
        !input.small && input.model.variants && input.user.model.variant
          ? input.model.variants[input.user.model.variant]
          : {}
      const base = input.small
        ? ProviderTransform.smallOptions(input.model)
        : ProviderTransform.options({
            model: input.model,
            sessionID: input.sessionID,
            providerOptions: item.options,
          })
      const options: Record<string, any> = pipe(
        base,
        mergeDeep(input.model.options),
        mergeDeep(input.agent.options),
        mergeDeep(variant),
      )
      if (isOpenaiOauth) {
        options.instructions = system.join("\n")
      }

      const isWorkflow = language instanceof GitLabWorkflowLanguageModel
      const messages = isOpenaiOauth
        ? input.messages
        : isWorkflow
          ? input.messages
          : [
              ...system.map(
                (x): ModelMessage => ({
                  role: "system",
                  content: x,
                }),
              ),
              ...input.messages,
            ]

      const params = yield* plugin.trigger(
        "chat.params",
        {
          sessionID: input.sessionID,
          agent: input.agent.name,
          model: input.model,
          provider: item,
          message: input.user,
        },
        {
          temperature: input.model.capabilities.temperature
            ? (input.agent.temperature ?? ProviderTransform.temperature(input.model))
            : undefined,
          topP: input.agent.topP ?? ProviderTransform.topP(input.model),
          topK: ProviderTransform.topK(input.model),
          maxOutputTokens: ProviderTransform.maxOutputTokens(input.model),
          options,
        },
      )

      const { headers } = yield* plugin.trigger(
        "chat.headers",
        {
          sessionID: input.sessionID,
          agent: input.agent.name,
          model: input.model,
          provider: item,
          message: input.user,
        },
        {
          headers: {},
        },
      )

      const tools = resolveTools(input)

      // LiteLLM and some Anthropic proxies require the tools parameter to be present
      // when message history contains tool calls, even if no tools are being used.
      // Add a dummy tool that is never called to satisfy this validation.
      // This is enabled for:
      // 1. Providers with "litellm" in their ID or API ID (auto-detected)
      // 2. Providers with explicit "litellmProxy: true" option (opt-in for custom gateways)
      const isLiteLLMProxy =
        item.options?.["litellmProxy"] === true ||
        input.model.providerID.toLowerCase().includes("litellm") ||
        input.model.api.id.toLowerCase().includes("litellm")

      // LiteLLM/Bedrock rejects requests where the message history contains tool
      // calls but no tools param is present. When there are no active tools (e.g.
      // during compaction), inject a stub tool to satisfy the validation requirement.
      // The stub description explicitly tells the model not to call it.
      if (
        (isLiteLLMProxy || input.model.providerID.includes("github-copilot")) &&
        Object.keys(tools).length === 0 &&
        hasToolCalls(input.messages)
      ) {
        tools["_noop"] = tool({
          description: "Do not call this tool. It exists only for API compatibility and must never be invoked.",
          inputSchema: jsonSchema({
            type: "object",
            properties: {
              reason: { type: "string", description: "Unused" },
            },
          }),
          execute: async () => ({ output: "", title: "", metadata: {} }),
        })
      }

      // Wire up toolExecutor for DWS workflow models so that tool calls
      // from the workflow service are executed via opencode's tool system
      // and results sent back over the WebSocket.
      if (language instanceof GitLabWorkflowLanguageModel) {
        const workflowModel = language as GitLabWorkflowLanguageModel & {
          sessionID?: string
          sessionPreapprovedTools?: string[]
          approvalHandler?: (approvalTools: { name: string; args: string }[]) => Promise<{ approved: boolean }>
        }
        workflowModel.sessionID = input.sessionID
        workflowModel.systemPrompt = system.join("\n")
        workflowModel.toolExecutor = async (toolName, argsJson, _requestID) => {
          const t = tools[toolName]
          if (!t || !t.execute) {
            return { result: "", error: `Unknown tool: ${toolName}` }
          }
          try {
            const result = await t.execute!(JSON.parse(argsJson), {
              toolCallId: _requestID,
              messages: input.messages,
              abortSignal: input.abort,
            })
            const output = typeof result === "string" ? result : (result?.output ?? JSON.stringify(result))
            return {
              result: output,
              metadata: typeof result === "object" ? result?.metadata : undefined,
              title: typeof result === "object" ? result?.title : undefined,
            }
          } catch (e: any) {
            return { result: "", error: e.message ?? String(e) }
          }
        }

        const ruleset = Permission.merge(input.agent.permission ?? [], input.permission ?? [])
        workflowModel.sessionPreapprovedTools = Object.keys(tools).filter((name) => {
          const match = ruleset.findLast((rule) => Wildcard.match(name, rule.permission))
          return !match || match.action !== "ask"
        })

        const bridge = yield* EffectBridge.make()
        const approvedToolsForSession = new Set<string>()
        workflowModel.approvalHandler = Instance.bind(async (approvalTools) => {
          const uniqueNames = [...new Set(approvalTools.map((t: { name: string }) => t.name))] as string[]
          // Auto-approve tools that were already approved in this session
          // (prevents infinite approval loops for server-side MCP tools)
          if (uniqueNames.every((name) => approvedToolsForSession.has(name))) {
            return { approved: true }
          }

          const id = PermissionID.ascending()
          let unsub: (() => void) | undefined
          try {
            unsub = Bus.subscribe(Permission.Event.Replied, (evt) => {
              if (evt.properties.requestID === id) void evt.properties.reply
            })
            const toolPatterns = approvalTools.map((t: { name: string; args: string }) => {
              try {
                const parsed = JSON.parse(t.args) as Record<string, unknown>
                const title = (parsed?.title ?? parsed?.name ?? "") as string
                return title ? `${t.name}: ${title}` : t.name
              } catch {
                return t.name
              }
            })
            const uniquePatterns = [...new Set(toolPatterns)] as string[]
            await bridge.promise(
              perm.ask({
                id,
                sessionID: SessionID.make(input.sessionID),
                permission: "workflow_tool_approval",
                patterns: uniquePatterns,
                metadata: { tools: approvalTools },
                always: uniquePatterns,
                ruleset: [],
              }),
            )
            for (const name of uniqueNames) approvedToolsForSession.add(name)
            workflowModel.sessionPreapprovedTools = [...(workflowModel.sessionPreapprovedTools ?? []), ...uniqueNames]
            return { approved: true }
          } catch {
            return { approved: false }
          } finally {
            unsub?.()
          }
        })
      }

      const tracer = cfg.experimental?.openTelemetry
        ? Option.getOrUndefined(yield* Effect.serviceOption(OtelTracer.OtelTracer))
        : undefined
      const telemetryTracer = tracer
        ? new Proxy(tracer, {
            get(target, prop, receiver) {
              if (prop !== "startSpan") return Reflect.get(target, prop, receiver)
              return (...args: Parameters<typeof target.startSpan>) => {
                const span = target.startSpan(...args)
                span.setAttribute("session.id", input.sessionID)
                return span
              }
            },
          })
        : undefined

      return streamText({
        onError(error) {
          l.error("stream error", {
            error,
          })
        },
        async experimental_repairToolCall(failed) {
          const result = repairToolCall(failed, tools, l)
          l.info("repairToolCall result", { tool: result.toolName })
          return result
        },
        temperature: params.temperature,
        topP: params.topP,
        topK: params.topK,
        providerOptions: ProviderTransform.providerOptions(input.model, params.options),
        activeTools: Object.keys(tools).filter((x) => x !== "invalid"),
        tools,
        toolChoice: input.toolChoice,
        maxOutputTokens: params.maxOutputTokens,
        abortSignal: input.abort,
        headers: {
          ...(input.model.providerID.startsWith("opencode")
            ? {
                "x-opencode-project": Instance.project.id,
                "x-opencode-session": input.sessionID,
                "x-opencode-request": input.user.id,
                "x-opencode-client": Flag.OPENCODE_CLIENT,
              }
            : {
                "x-session-affinity": input.sessionID,
                ...(input.parentSessionID ? { "x-parent-session-id": input.parentSessionID } : {}),
                "User-Agent": `opencode/${InstallationVersion}`,
              }),
          ...input.model.headers,
          ...headers,
        },
        maxRetries: input.retries ?? 0,
        messages,
        model: wrapLanguageModel({
          model: language,
          middleware: [
            {
              specificationVersion: "v3" as const,
              async transformParams(args) {
                if (args.type === "stream") {
                  // @ts-expect-error
                  args.params.prompt = ProviderTransform.message(args.params.prompt, input.model, options)
                }
                return args.params
              },
            },
          ],
        }),
        experimental_telemetry: {
          isEnabled: cfg.experimental?.openTelemetry,
          functionId: "session.llm",
          tracer: telemetryTracer,
          metadata: {
            userId: cfg.username ?? "unknown",
            sessionId: input.sessionID,
          },
        },
      })
    })

    const stream: Interface["stream"] = (input) =>
      Stream.scoped(
        Stream.unwrap(
          Effect.gen(function* () {
            const ctrl = yield* Effect.acquireRelease(
              Effect.sync(() => new AbortController()),
              (ctrl) => Effect.sync(() => ctrl.abort()),
            )

            const result = yield* run({ ...input, abort: ctrl.signal })

            return Stream.fromAsyncIterable(result.fullStream, (e) => (e instanceof Error ? e : new Error(String(e))))
          }),
        ),
      )

    return Service.of({ stream })
  }),
)

export const layer = live.pipe(Layer.provide(Permission.defaultLayer))

export const defaultLayer = Layer.suspend(() =>
  layer.pipe(
    Layer.provide(Auth.defaultLayer),
    Layer.provide(Config.defaultLayer),
    Layer.provide(Provider.defaultLayer),
    Layer.provide(Plugin.defaultLayer),
  ),
)

function resolveTools(input: Pick<StreamInput, "tools" | "agent" | "permission" | "user">) {
  const disabled = Permission.disabled(
    Object.keys(input.tools),
    Permission.merge(input.agent.permission, input.permission ?? []),
  )
  return Record.filter(input.tools, (_, k) => input.user.tools?.[k] !== false && !disabled.has(k))
}

function normalizeFilePath(input: Record<string, unknown>, toolName: string): string | null {
  const lower = toolName.toLowerCase()
  if (!["write", "read", "edit", "multiedit", "lsp"].includes(lower)) return null
  if (typeof input.filePath === "string" && input.filePath.trim()) return null
  const filePath = [input.filepath, input.path, input.file, input.filename]
    .map((v) => (typeof v === "string" ? v.trim() : undefined))
    .find((x) => x)
  if (!filePath) return null
  const fixed: Record<string, unknown> = { ...input, filePath }
  delete fixed.filepath
  if (lower === "write") delete fixed.path
  delete fixed.file
  delete fixed.filename
  return JSON.stringify(fixed)
}

function repairToolCall(
  failed: {
    toolCall: { type: "tool-call"; toolName: string; input: string; toolCallId: string }
    error: Error
  },
  tools: Record<string, unknown>,
  log: ReturnType<typeof Log.create>,
) {
  const name = failed.toolCall.toolName
  const lower = name.toLowerCase()

  if (lower !== name && tools[lower]) {
    log.info("repairing tool call", { tool: name, repaired: lower })
    return { ...failed.toolCall, toolName: lower }
  }

  const resolvedName = tools[lower] ? lower : name
  const isInvalidInput = InvalidToolInputError.isInstance(failed.error)
  const hasJsonError =
    isInvalidInput ||
    (failed.error?.message &&
      (failed.error.message.includes("JSON") ||
        failed.error.message.includes("parse") ||
        failed.error.message.includes("Unterminated")))

  if (hasJsonError && tools[resolvedName]) {
    const toolInput = isInvalidInput ? (failed.error as InvalidToolInputError).toolInput : failed.toolCall.input
    const repaired = repairToolCallJson(toolInput, log, resolvedName)
    if (repaired !== undefined) {
      try {
        const parsed = JSON.parse(repaired)
        if (typeof parsed === "object" && parsed !== null) {
          const normalized = normalizeFilePath(parsed, resolvedName)
          if (normalized) {
            log.info("normalized filePath for tool", { tool: resolvedName })
            return { ...failed.toolCall, toolName: resolvedName, input: normalized }
          }
        }
      } catch {}
      log.info("repaired malformed tool call JSON", { tool: resolvedName })
      return { ...failed.toolCall, toolName: resolvedName, input: repaired }
    }

    try {
      const parsed = JSON.parse(toolInput)
      if (typeof parsed === "object" && parsed !== null) {
        const normalized = normalizeFilePath(parsed, resolvedName)
        if (normalized) {
          log.info("normalized filePath for tool", { tool: resolvedName })
          return { ...failed.toolCall, toolName: resolvedName, input: normalized }
        }
      }
    } catch {}
  }

  const isNoSuchTool = NoSuchToolError.isInstance(failed.error)
  const msg = isNoSuchTool
    ? `Tool "${name}" is not available. Please use one of the tools provided to you.`
    : `Tool "${name}" was called with invalid arguments. Please review the tool schema and retry.`

  return {
    ...failed.toolCall,
    input: JSON.stringify({ tool: name, error: msg }),
    toolName: "invalid",
  }
}

function repairToolCallJson(input: string, log: ReturnType<typeof Log.create>, toolName: string): string | undefined {
  try {
    JSON.parse(input)
    return undefined
  } catch {}

  const trimmed = input.trimStart()
  if (!trimmed.startsWith("{") && !trimmed.startsWith("[") && !trimmed.startsWith("`")) return undefined

  // Strategy 1: Find first complete JSON object (handles duplicated JSON)
  let balance = 0
  let inStr = false
  let esc = false
  for (let i = 0; i < input.length; i++) {
    const ch = input[i]
    if (esc) { esc = false; continue }
    if (ch === "\\") { esc = true; continue }
    if (ch === '"') { inStr = !inStr; continue }
    if (inStr) continue
    if (ch === "{" || ch === "[") balance++
    else if (ch === "}" || ch === "]") balance--
    if (balance === 0 && (ch === "}" || ch === "]")) {
      const candidate = input.substring(0, i + 1)
      try {
        JSON.parse(candidate)
        if (i < input.length - 1) {
          log.info("repaired JSON by extracting first complete object", { tool: toolName })
          return candidate
        }
      } catch {}
    }
  }

  // Strategy 2: Duplicated JSON where first object is missing closing brace
  // e.g. {"query":"x","max":10{"query":"x","max":10}
  balance = 0
  inStr = false
  esc = false
  for (let i = 0; i < input.length; i++) {
    const ch = input[i]
    if (esc) { esc = false; continue }
    if (ch === "\\") { esc = true; continue }
    if (ch === '"') { inStr = !inStr; continue }
    if (inStr) continue
    if (ch === "{" || ch === "[") {
      balance++
      if (balance === 2 && ch === "{") {
        const candidate = input.substring(0, i) + "}"
        try {
          JSON.parse(candidate)
          log.info("repaired JSON by closing duplicated object", { tool: toolName })
          return candidate
        } catch {}
      }
    }
    if (ch === "}" || ch === "]") balance--
  }

  // Strategy 3: Truncated JSON — close unterminated strings and open brackets
  let repaired = input
  let esc2 = false
  let inStr2 = false
  const openStack: string[] = []
  for (const ch of repaired) {
    if (esc2) { esc2 = false; continue }
    if (ch === "\\") { esc2 = true; continue }
    if (ch === '"') { inStr2 = !inStr2; continue }
    if (inStr2) continue
    if (ch === "{" || ch === "[") openStack.push(ch)
    if (ch === "}" && openStack.length > 0 && openStack[openStack.length - 1] === "{") openStack.pop()
    if (ch === "]" && openStack.length > 0 && openStack[openStack.length - 1] === "[") openStack.pop()
  }
  if (inStr2) repaired += '"'
  for (let i = openStack.length - 1; i >= 0; i--) {
    repaired += openStack[i] === "{" ? "}" : "]"
  }

  try {
    const parsed = JSON.parse(repaired)
    if (typeof parsed === "object" && parsed !== null && Object.keys(parsed).length === 0 && trimmed.length <= 2) return undefined
    log.info("repaired JSON by closing truncated structure", { tool: toolName })
    return repaired
  } catch {}

  return undefined
}

// Check if messages contain any tool-call content
// Used to determine if a dummy tool should be added for LiteLLM proxy compatibility
export function hasToolCalls(messages: ModelMessage[]): boolean {
  for (const msg of messages) {
    if (!Array.isArray(msg.content)) continue
    for (const part of msg.content) {
      if (part.type === "tool-call" || part.type === "tool-result") return true
    }
  }
  return false
}

export { repairToolCallJson, normalizeFilePath, repairToolCall }

export * as LLM from "./llm"

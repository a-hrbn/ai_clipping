# ralph_stream_render.jq — render `claude -p --output-format stream-json --verbose`
# into a compact, Claude-Code-style transcript for the terminal.
#
# Invoked as:  jq -rR --unbuffered --argjson head_n 8 -f ralph_stream_render.jq
#   -R : read one RAW line per input (so stray non-JSON stderr lines don't crash
#        the parse — they're passed through verbatim).
#   -r : raw string output (no JSON quoting).
#   $head_n : how many lines of a tool result to show before truncating.
#
# ANSI colors are emitted unconditionally; the caller strips them (sed) for the
# plain-text .log copy.

def esc: "[";
def reset: esc + "0m";
def dim(s):    esc + "2m"  + s + reset;
def bold(s):   esc + "1m"  + s + reset;
def cyan(s):   esc + "36m" + s + reset;
def green(s):  esc + "32m" + s + reset;
def red(s):    esc + "31m" + s + reset;

def indent: split("\n") | map("    " + .) | join("\n");

def head_lines($n):
  . as $s
  | ($s | split("\n")) as $ls
  | if ($ls | length) > $n
    then (($ls[0:$n]) | join("\n")) + "\n" + dim("    … +" + (($ls | length) - $n | tostring) + " Zeilen")
    else $s end;

# Concise one-line summary of a tool call's input.
def tool_line($name; $input):
  ( if $name == "Bash" then ($input.command // $input.description // "")
    elif ($name | test("^(Edit|Write|Read|NotebookEdit)$")) then ($input.file_path // ($input | tojson))
    elif ($name | test("^(Grep|Glob)$")) then ($input.pattern // ($input | tojson))
    elif $name == "Task" then ($input.description // "")
    else ($input | tojson) end ) as $arg
  | "⏺ " + bold($name) + cyan("(" + ($arg | gsub("\n"; " ") | .[0:200]) + ")");

# tool_result content is either a plain string or an array of content blocks.
def result_text:
  if   (.content | type) == "string" then .content
  elif (.content | type) == "array"  then ([.content[]? | (.text // "")] | join("\n"))
  else (.content | tojson) end;

# --- main: one raw input line at a time -------------------------------------
(try fromjson catch null) as $e
| if $e == null then (. | select(length > 0))          # non-JSON (stderr) → verbatim
  else
    $e
    | if .type == "system" and .subtype == "init" then
        dim("◇ session " + (.model // "?") + "  cwd=" + (.cwd // "?") + "  mode=" + (.permissionMode // "?"))

      elif .type == "assistant" then
        ( [ .message.content[]?
            | if   .type == "text"              then (.text // "" | select(length > 0))
              elif .type == "thinking"          then dim("✻ Thinking…")
              elif .type == "redacted_thinking" then dim("✻ Thinking… (redacted)")
              elif .type == "tool_use"          then tool_line(.name; (.input // {}))
              else empty end ]
          | join("\n") ) | select(length > 0)

      elif .type == "user" then
        ( [ .message.content[]?
            | if .type == "tool_result" then
                ( (if .is_error == true then red("  ⎿ Fehler:") else dim("  ⎿") end)
                  + "\n" + (result_text | head_lines($head_n) | indent) )
              else empty end ]
          | join("\n") ) | select(length > 0)

      elif .type == "result" then
        ( (if .is_error then red("✗") else green("✓") end)
          + " " + bold("Ergebnis")
          + dim("  turns=" + ((.num_turns // 0) | tostring)
                + "  " + (((.duration_ms // 0) / 1000) | floor | tostring) + "s"
                + "  $" + ((.total_cost_usd // 0) | tostring)) )

      else empty end
  end

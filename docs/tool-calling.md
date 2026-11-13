# Tool calling

Both model families can call tools, in the format each was trained on: Gemma 4 declares them in
its system turn (`<|tool>declaration:...<tool|>`) and calls `<|tool_call>call:NAME{...}<tool_call|>`;
Qwen 3.5 / 3.8 lists JSON schemas in `<tools>` and answers with `<tool_call><function=...>`.
`src/tools.rs` formats the declarations, parses the calls and feeds results back the way each
template does, and the rendered prompts are byte-identical to `transformers`'
`apply_chat_template(..., tools=...)` for both families.

On the command line `--tools` enables two built-in tools, the local date and time and an exact
calculator:

```sh
./target/release/oxinfer --model google/gemma-4-E4B-it --quant q4 --tools -p "What is 1234.5 * 6789 - 2^20?"
# [tool calculate({"expression":"1234.5 * 6789 - 2^20"}) -> 7332444.5]
# The result of 1234.5 x 6789 - 2^20 is **7332444.5**.
```

In the browser the model gets `get_datetime` and `run_javascript`, which runs the code it writes in
a Web Worker with the network and storage APIs removed and a 5 s limit (hardening, not a security
boundary). Its system prompt describes where it runs: browser, operating system, language, time
zone, local time, screen, CPU cores and GPU. Both can be changed under Settings.

### mollama

This wants to be an Ollama replacement for a specific purpose. As of this
writing, it is a complete inference engine that accepts JSON prompts and
streams JSON fragments. It works, but it does not do anything useful. It
also needs some clean-ups, proper handling of long prompts (without seg
faults), and some efficiency tweaks. But it does serve the purpose of being
educational.

It was an attempt to build a small inference engine over a weekend. It took
twice that much time (check the commits). But it is still an impressive
feat and I have learned a lot.

You want to wait until I actually make it work the way I want (load GPT2-oss
weights, CPU parallelized with POSIX or OpenMP threading, a proper REPL
front end).

IN 2026/03/17


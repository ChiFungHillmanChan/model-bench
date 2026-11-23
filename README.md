# model-bench

Private model benchmark and test harness on Inspect AI. See
`docs/superpowers/specs/2026-11-22-model-bench-design.md`.

```
uv sync
uv run bench --help
```

## Private tasks

Tasks you do not want published go in `tasks/private/` (ignored by git) and are registered in
`tasks/private/registry.yaml`; `bench` merges that file into `tasks/registry.yaml` when it exists.
The dashboard (`reports/dashboard.html`) shows scores, rates and timings only, never question text
or answers.

## License

MIT, see [LICENSE](LICENSE). Contributors follow the [Code of Conduct](CODE_OF_CONDUCT.md).

Benchmark datasets are downloaded at run time from their own sources and keep their own licenses and
terms (some, such as GPQA, ask that examples are not republished). Do not commit run logs or
`results.duckdb`: they contain benchmark questions and answers.


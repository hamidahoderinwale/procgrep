# Security policy

## Supported versions

procgrep is pre-1.0. Only the latest `0.1.x` release receives
security fixes. Versions older than the latest minor are not
supported.

| Version | Supported |
|---|---|
| 0.1.x (latest) | yes |
| older | no |

## Reporting a vulnerability

If you discover a security issue, please do not open a public issue.
Email the maintainer directly (see the `authors` field in
`CITATION.cff` for contact) with:

- A description of the issue.
- A minimal reproducible example.
- The procgrep version, Python version, and operating system.
- Any context on impact.

You should expect an acknowledgement within a few business days. If
the issue is confirmed, a fix will land on `main` and a patch
release will be tagged.

## Scope

procgrep is a post-hoc analysis library that reads trace files. It
does not execute agent code, call models, or send data over the
network. The relevant security surface is therefore narrow:

- Parsing of trace JSONL: malformed input should produce a clear
  error, not an arbitrary crash or unbounded resource use.
- Parsing of YAML rule files: same standard.
- Regex compilation in the pattern matcher: pathological patterns
  should be the user's responsibility, not a procgrep vulnerability,
  unless they cause a hang on otherwise-valid input.

Out of scope: anything related to the agents whose traces you are
analyzing, or to the model providers those agents use.

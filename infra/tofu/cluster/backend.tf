# D2: tofu state is durable state plane inventory, not local disk (R2, "tofu-state/" prefix per
# infra/README.md's bucket layout doc, outside Hetzner by construction so state survives a torn down
# cluster). Deliberately a PARTIAL backend block: no bucket/key/region/endpoint literal lives in this
# committed file, because an R2 endpoint URL encodes the operator's own Cloudflare account id, and
# this repo's own discipline (infra/README.md's SOPS section, .env.fastlane) keeps operator specific,
# non secret identifiers out of committed files the same way it keeps real secrets out.
#
# infra/scripts/burst-up.sh supplies the missing attributes at `tofu init` time via a generated
# `-backend-config=<tmpfile>.hcl` (bucket, key, region "auto", the R2 endpoint from ATLAS_R2_ENDPOINT,
# use_path_style true; credentials come from AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY in the
# environment, never written to any file), then deletes the temp file. This is exactly why `tofu
# validate` in this task's own acceptance bar always runs as `tofu init -backend=false && tofu
# validate`: partial config with zero attributes supplied still parses and type checks with no
# backend initialized at all, which is what a credential free, state free validate needs.
terraform {
  backend "s3" {}
}

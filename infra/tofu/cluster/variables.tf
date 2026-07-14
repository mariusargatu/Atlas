# Every variable here is supplied by infra/scripts/burst-up.sh (TF_VAR_* env vars, populated only
# after that script's own credential gate passes) or left at a safe default; none has a real value
# committed anywhere in this repo. `tofu validate` treats a required variable with no default as an
# unknown value and validates structurally without it (verified live: `env -u HCLOUD_TOKEN ... tofu
# validate` passes with none of these set), which is the actual mechanism "tofu validate clean with
# credentials absent" rests on, not a special case carved out for this task.

variable "hcloud_token" {
  description = "Hetzner Cloud API token (project scoped). From $HCLOUD_TOKEN; never a literal here."
  type        = string
  sensitive   = true
}

variable "ssh_public_key" {
  description = <<-EOT
    Single line OpenSSH public key content for node access, read by infra/scripts/burst-up.sh from
    ~/.config/atlas-hetzner/id_ed25519.pub (documented in infra/README.md) and passed through as
    TF_VAR_ssh_public_key. Deliberately not read here via tofu's own `file()`: that would make
    `tofu validate` depend on a file existing on disk, which breaks validate in a hermetic checkout
    that has never provisioned an operator SSH keypair.
  EOT
  type        = string
}

variable "ssh_private_key" {
  description = <<-EOT
    SSH private key content, or null (the default) to authenticate via a running ssh-agent instead
    (kube-hetzner's own documented safer path: `ssh-add ~/.config/atlas-hetzner/id_ed25519`). Private
    key material is deliberately never routed through an env var or a committed file by this task's
    own scripts; null is the only value infra/scripts/burst-up.sh ever passes.
  EOT
  type        = string
  sensitive   = true
  default     = null
}

variable "cluster_name" {
  description = <<-EOT
    Hetzner resource name prefix AND the "cluster" hcloud label value kube-hetzner stamps on every
    resource it creates (verified against the module's own locals.tf: `labels = { provisioner =
    "terraform", engine = ..., cluster = var.cluster_name }`). infra/scripts/hcloud-orphans.sh and
    .github/workflows/janitor.yml both filter on `cluster=<this value>` -- a label selector, never a
    scan of the whole hcloud account -- which is precisely what keeps the standing atlas-fastlane box
    (id 152778751, never created by this module, never carrying this label) out of scope by
    construction, not by an incidental name difference.
  EOT
  type        = string
  default     = "atlas-burst"
}

variable "base_domain" {
  description = <<-EOT
    The burst tier's base domain (e.g. "atlas.example.com"), used for the wildcard cert's DNS names
    and reverse DNS. From $ATLAS_BURST_DOMAIN (infra/README.md's "Burst DNS" section); empty string
    (kube-hetzner's own default) is valid and simply skips reverse DNS / cert wiring at the tofu
    layer -- infra/charts/atlas-cert (helmfile, burst only) is what actually issues the cert, gated
    on the same variable via `environments/burst/values.yaml`'s own `ref+envsubst://$ATLAS_BURST_DOMAIN`.
  EOT
  type        = string
  default     = ""
}

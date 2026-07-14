# SP5 task 5: the Hetzner burst tier's cluster shape. This directory's own contract, kept small on
# purpose so a future sibling module (AWS, GCP, another cloud) can satisfy the same contract without
# touching anything above it (infra/README.md's own "Cloud portability boundary" section names this
# split): produce a k8s cluster + a kubeconfig + node labels. Nothing here is applied by this task;
# `tofu validate` (no credentials, no state, no network call to Hetzner) is the acceptance bar,
# proven with `tofu init -backend=false && tofu validate` in a clean environment.
terraform {
  required_version = ">= 1.11.0"
  required_providers {
    hcloud = {
      source  = "hetznercloud/hcloud"
      version = "1.66.1" # pinned (D26: never a floating alias), matching the module's own exact pin discipline
    }
  }
}

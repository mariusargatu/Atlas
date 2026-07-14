#!/usr/bin/env bash
# SP5 task 5: the shared burst tier orphan check, called from BOTH infra/scripts/burst-destroy.sh (a
# post destroy sanity check) and .github/workflows/janitor.yml (the weekly standing check), so the
# same definition of "orphan" is never duplicated between the two -- exactly the instruction this
# task carries forward from the render test's own "burst leak guard" precedent (2047e1a): one guard,
# never a second copy that could quietly drift from the first.
#
# Label based, never a scan of "any resource in the account": every resource kube-hetzner's own
# infra/tofu/cluster module creates carries a "cluster=<cluster_name>" hcloud label (verified against
# the module's own source, not assumed -- locals.tf: `labels = { provisioner = "terraform", engine =
# ..., cluster = var.cluster_name }`, merged onto every hcloud_server/hcloud_load_balancer/
# hcloud_network/hcloud_firewall/hcloud_volume/hcloud_placement_group it manages). cluster_name
# defaults to "atlas-burst" (infra/tofu/cluster/variables.tf), a name chosen specifically so it can
# never collide with the user's own standing atlas-fastlane box (id 152778751, hand created, never
# run through this module, never carrying this label). This is what "the janitor's orphan definition
# should be label based... precisely so the user's fastlane box is never flagged by design" means in
# code, not just in a comment: fastlane is out of scope because nothing here ever queries by name or
# lists the whole account, only by a label this script never applies to anything itself.
#
# Precondition: HCLOUD_TOKEN must already be exported by the caller. This script does not gate on it
# itself -- infra/scripts/burst-up.sh / burst-destroy.sh and janitor.yml each own their own credential
# gate message, worded for their own context (burst-up/destroy fail hard on a missing token; the
# janitor treats an absent token as a neutral skip, see that workflow's own comment on why).
set -euo pipefail

CLUSTER_LABEL="${ATLAS_BURST_CLUSTER_LABEL:-cluster=atlas-burst}"

# Every hcloud resource type infra/tofu/cluster's kube-hetzner module can create (servers, the
# private network, the firewall, the load balancer, any attached volumes, placement groups, floating
# IPs). Listed explicitly, not via some "all resource types" introspection (hcloud's own `all list`
# command exists but combines every type into one call with a JSON envelope this task could not
# verify live, no credentials being available to this task by its own hard safety rules -- looping
# the well documented per type `list -l <selector> -o json` commands instead keeps this script's own
# correctness checkable by reading `hcloud <type> list --help` directly, not by trusting an assumed
# schema).
RESOURCE_TYPES=(server load-balancer network firewall volume placement-group floating-ip)

found_any=0
for type in "${RESOURCE_TYPES[@]}"; do
  rows="$(hcloud "${type}" list -l "${CLUSTER_LABEL}" -o noheader -o columns=id,name)"
  if [[ -n "${rows}" ]]; then
    found_any=1
    echo "ORPHAN: ${type} resources still carry label '${CLUSTER_LABEL}':"
    echo "${rows}" | sed 's/^/  /'
  fi
done

if [[ "${found_any}" -eq 1 ]]; then
  echo "one or more hcloud resources labeled '${CLUSTER_LABEL}' still exist. If a burst run is genuinely active right now, this is an expected false positive (HLD section 7.3's own accepted trade off for a rare, ephemeral tier with no cheap way to distinguish 'intentionally up' from 'leaked'); otherwise this is a real leak outside tofu state. Investigate with 'hcloud <type> list -l ${CLUSTER_LABEL}' and clean up manually, or rerun 'task burst:destroy' if tofu state still tracks it." >&2
  exit 1
fi

echo "clean: no hcloud resources labeled '${CLUSTER_LABEL}' found."

# infra/scripts/burst-up.sh reads these after a real apply (never invoked by this task): the
# kubeconfig for `helmfile -e burst sync`/`kubectl`, and the load balancer's public IPv4 for the
# static DNS instructions this task's own D3 revised decision calls for ("static DNS record to the
# LB; external-dns dropped" -- no automation writes a DNS record, an operator does, once, using this
# printed value).
output "kubeconfig" {
  description = "The burst cluster's kubeconfig (contains a client certificate; sensitive)."
  value       = module.kube_hetzner.kubeconfig
  sensitive   = true
}

output "load_balancer_public_ipv4" {
  description = "The Hetzner Load Balancer's public IPv4, what the operator points the static DNS A record at (D3: external-dns dropped, this is a printed value, not an automated write)."
  value       = module.kube_hetzner.ingress_public_ipv4
}

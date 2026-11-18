# Security

Please report vulnerabilities privately through GitHub's "Report a vulnerability" (Security tab) rather
than in a public issue.

Dependencies are kept current by Dependabot, and CI runs `pip-audit` against the resolved environment on
every push. Minimum versions exclude releases with known advisories (for example LightGBM >= 4.6.0 for
CVE-2024-43598). `trexboost` never loads models or data from files or the network; it only fits on arrays
you pass in.

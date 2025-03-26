Some comments on the runs performed
- `injection_1` -- `injection_10`: Good recovery
    - `injection_5`: Distance is a bit off, some weird features visible.
    - `injection_7`: Distance is railing because the injected value is really at the edge of the prior. This needs to be fixed later on by differentiating between priors for generating injections and for sampling.
    - `injection_10`: Interesting: high q so flat Lambda2, but 
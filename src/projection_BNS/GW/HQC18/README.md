Some comments on the runs performed
- `injection_1` -- `injection_10`: Good recovery. There are a few for which I don't want to use for testing for now.
    - `injection_5`: Distance is a bit off, some weird features visible.
    - `injection_7`: Distance is railing because the injected value is really at the edge of the prior. This needs to be fixed later on by differentiating between priors for generating injections and for sampling.
- `injection_11` -- `injection_30`: Lots of high chirp masses. It is definitely time to start generating with a uniform component masses prior.
    - `injection_13`: Lambdas are weird.
    - `injection_23`: Good one!
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and release versions follow [PEP 440](https://peps.python.org/pep-0440/).

## [Unreleased]

## [0.1.0] - 2026-08-03

This is the first stable release of aspire-bilby. See the prerelease entries
below for the development history leading to this release.

### Added

- Add configurable Aspire log output streams ([#22]).

### Changed

- Require Aspire 0.1.0 or later.
- Require MiniPCN 0.2.0 or later when using the `minipcn` extra.

## [0.1.0a8] - 2026-06-23

### Added

- Validate explicitly specified initial-sample parameters ([#18]).
- Support the `spawn` and `forkserver` multiprocessing start methods ([#21]).

### Changed

- Improve conversion of Aspire outputs into Bilby results, including posterior
  samples, weights, evidence estimates, likelihood evaluations, and sampling
  time ([#16]).
- Require Python 3.11 or later ([#20]).

### Fixed

- Pass sampler keyword arguments correctly in the SMC example ([#19]).

## [0.1.0a7] - 2026-02-25

### Added

- Add an adaptive SMC example ([#17]).

### Fixed

- Only add the log evidence to a Bilby result when it is available ([#15]).

## [0.1.0a6] - 2026-02-20

### Changed

- Add compatibility with Bilby 2.7 ([#14]).

## [0.1.0a5] - 2026-02-16

### Changed

- Update sample conversion for Aspire's `to_dataframe` API ([#12]).

### Fixed

- Always define posterior samples when converting an Aspire result to a Bilby
  result ([#11]).
- Use Bilby's supported random-number generator API ([#13]).

## [0.1.0a4] - 2025-12-16

### Fixed

- Restore checkpointed runs correctly when using Bilby Pipe ([#9]).

## [0.1.0a3] - 2025-12-16

### Added

- Add initial hosted documentation ([#7]).
- Add checkpointing and resume support ([#8]).

## [0.1.0a2] - 2025-11-03

### Added

- Support regular expressions when selecting parameters that should be sampled
  from updated priors ([#6]).

## [0.1.0a1] - 2025-09-23

### Added

- Add the initial Aspire sampler plugin for Bilby.
- Add conversion of Bilby likelihoods and priors into Aspire-compatible
  functions.
- Add initialization from Bilby results and direct sampling from Bilby priors.
- Add support for Bilby Pipe, multiprocessing, and configurable logging.
- Add sampling of parameters missing from an existing result.
- Add likelihood dtype selection, final-sample controls, parameter conversion,
  and diagnostic plotting ([#1]).
- Add the initial test suite ([#3]).
- Add continuous-integration and publishing workflows ([#4]).
- Add initial README documentation ([#5]).

### Changed

- Rename the integration from Poppy to Aspire ([#2]).

[Unreleased]: https://github.com/mj-will/aspire-bilby/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mj-will/aspire-bilby/compare/v0.1.0a8...v0.1.0
[0.1.0a8]: https://github.com/mj-will/aspire-bilby/compare/v0.1.0a7...v0.1.0a8
[0.1.0a7]: https://github.com/mj-will/aspire-bilby/compare/v0.1.0a6...v0.1.0a7
[0.1.0a6]: https://github.com/mj-will/aspire-bilby/compare/v0.1.0a5...v0.1.0a6
[0.1.0a5]: https://github.com/mj-will/aspire-bilby/compare/v0.1.0a4...v0.1.0a5
[0.1.0a4]: https://github.com/mj-will/aspire-bilby/compare/v0.1.0a3...v0.1.0a4
[0.1.0a3]: https://github.com/mj-will/aspire-bilby/compare/v0.1.0a2...v0.1.0a3
[0.1.0a2]: https://github.com/mj-will/aspire-bilby/compare/v0.1.0a1...v0.1.0a2
[0.1.0a1]: https://github.com/mj-will/aspire-bilby/releases/tag/v0.1.0a1
[#1]: https://github.com/mj-will/aspire-bilby/pull/1
[#2]: https://github.com/mj-will/aspire-bilby/pull/2
[#3]: https://github.com/mj-will/aspire-bilby/pull/3
[#4]: https://github.com/mj-will/aspire-bilby/pull/4
[#5]: https://github.com/mj-will/aspire-bilby/pull/5
[#6]: https://github.com/mj-will/aspire-bilby/pull/6
[#7]: https://github.com/mj-will/aspire-bilby/pull/7
[#8]: https://github.com/mj-will/aspire-bilby/pull/8
[#9]: https://github.com/mj-will/aspire-bilby/pull/9
[#11]: https://github.com/mj-will/aspire-bilby/pull/11
[#12]: https://github.com/mj-will/aspire-bilby/pull/12
[#13]: https://github.com/mj-will/aspire-bilby/pull/13
[#14]: https://github.com/mj-will/aspire-bilby/pull/14
[#15]: https://github.com/mj-will/aspire-bilby/pull/15
[#16]: https://github.com/mj-will/aspire-bilby/pull/16
[#17]: https://github.com/mj-will/aspire-bilby/pull/17
[#18]: https://github.com/mj-will/aspire-bilby/pull/18
[#19]: https://github.com/mj-will/aspire-bilby/pull/19
[#20]: https://github.com/mj-will/aspire-bilby/pull/20
[#21]: https://github.com/mj-will/aspire-bilby/pull/21
[#22]: https://github.com/mj-will/aspire-bilby/pull/22

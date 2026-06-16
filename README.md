# FusDB

`fusdb` is a personal project for collecting, organizing and making easily reusable functions and reactor data commonly used in nuclear fusion reactor anlysis.

The project started from a simple practical question I had during the Christmas holidays of 2025
> "Why can't I have a database for all the useful functions used in nuclear fusion reactor studies?"
I initially began by collecting and structuring functions that I found myself reusing across different analyses, heavily inspired by [cfspopcon](https://github.com/cfs-energy/cfspopcon).  

Later, during my PhD work, I often had to search through many papers to collect fusion reactor parameters, compare assumptions, and check consistency between sources. This process was time-consuming and difficult to reproduce. So I asked myself:
> "Could I automate the verification of a reactor operating scenario?"
This pushed me to include a way to _verify_ and possibly _reconcile_ a set of parameters defining an operating scenario, similarly to what [PROCESS](https://github.com/ukaea/PROCESS) does.

In the meantime, I was reorganizing my notes on the plasma physics courses I did during my master and PhD, and I wanted to have them all in one place.
> "Can FusDB include the relevant plasma physics background together with the relations it uses?"

FusDB was born from the need to make essential relations and data easier to reuse in different contexts. Usually functions and data are embedded inside individual scripts inside a larger framework. This makes them difficult to discover, inspect, or use in a different contexts.  
FusDB tries to address this by treating _relations_ as independent objects, that can be exported and used in external code. 

Over time, the project evolved in three directions:

1. a reusable collection of relations, relevant for nuclear fusion applications;
2. a framework for reproducing and checking reactor operating scenarios;
3. a lightweight knowledge base containing definitions, explanations, assumptions, and references for the quantities and relations it uses.


## Installation

```bash
pip install -e .
```

For documentation development:

```bash
pip install -e ".[docs]"
mkdocs serve -f docs/mkdocs.yml
```

## Where To Look Next

- Documentation site: [https://alessmor.github.io/fusdb/](https://alessmor.github.io/fusdb/)
- Source documentation: [https://alessmor.github.io/fusdb/code_docs/](https://alessmor.github.io/fusdb/code_docs/)
- Explore reactors: [https://alessmor.github.io/fusdb/getting_started/reactors/](https://alessmor.github.io/fusdb/getting_started/reactors/)

## Third-party Notices

This repository contains code adapted from open-source projects.
The relevant license notices, source code references are listed below.

### Code adapted from cfspopcon  

- Repository: https://github.com/cfs-energy/cfspopcon  
- Copyright notice: Copyright (c) 2023 Commonwealth Fusion Systems
- License: MIT License
- Identifier used in the code: # Adapted from cfspopcon; see README.md section "Third-party Notices".

### Code adapted from PROCESS

- Repository: https://github.com/ukaea/PROCESS
- Copyright notice: Copyright (c) 2023 United Kingdom Atomic Energy Authority
- License: MIT License
- Identifier used in the code: # Adapted from PROCESS; see README.md section "Third-party Notices".

## Use of generative AI disclosure:

FusDB is a personal project, that goes beyond the activities carried out during my PhD.  
For this reason, I used AI tools to help me during code and drafting, refactoring, documentation and testing.  

Nonetheless, FusDB was reviewed and accepted manually. This is especially true for all physics relations that, whenever possible, are also accompanied by a published reference.

## Applicability of FusDB:

FusDB is meant as a practical tool, not as a scientific source. Results should be checked before being used in analysis.

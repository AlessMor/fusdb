---
status: Draft
bibliography: ../../../bibliography/bibliography.bib
---

# Electrons, Ions, Neutrals and Molecular Interactions

Based on chapter 4 of [@militello_boundary_2022].  
For simplicity, only hydrogenic species are considered.  
[Fusion reactions](../0-fusion_plasmas/cross_sections_reactivities.md) and reactions concerning [impurities](impurities.md) are excluded.  

## Abstract

!!! abstract "Definition / Summary"
    This page summarizes the main elementary interactions among electrons, ions, atoms, molecules, photons, and surfaces in hydrogenic fusion edge plasmas. Reactions are written as named one-way processes. When a commonly used inverse process exists, it is listed separately and cross-referenced.

## Conventions

!!! note "Reaction-arrow convention"
    A single arrow, `\rightarrow`, is used for the named process in its usual direction.  
    The inverse process is listed separately when relevant.  
    Elastic scattering processes are treated as self-inverse.

!!! note "Notation"
    \(H\) denotes a hydrogenic atom.  
    \(H^+\) denotes a hydrogenic ion.  
    \(H^-\) denotes a negative hydrogenic ion.  
    \(H^*\) or \(H(n)\) denotes an electronically excited hydrogenic atom.  
    \(H_2(v,J)\) denotes a molecular hydrogenic species in vibrational state \(v\) and rotational state \(J\).  
    \(h\nu\) denotes a photon.

---

## Electron-Electron Interactions

### Elastic Coulomb Collision

Usually considered in transport or collisional operators.

$$
e + e \rightarrow e + e
$$

 

---

## Electron-Ion Interactions

### Elastic Coulomb Collision

Usually considered in transport or collisional operators.

$$
e + H^+ \rightarrow e + H^+
$$

 

### Radiative Recombination

$$
H^+ + e \rightarrow H(n) + h\nu
$$

The inverse process is [Photo-Ionization](#photo-ionization).

### Three-Body Recombination

$$
H^+ + e + e \rightarrow H(n) + e
$$

The inverse process is [Collisional Ionization](#collisional-ionization).

### Bremsstrahlung

$$
e + H^+ \rightarrow e + H^+ + h\nu
$$

The inverse process is [Inverse Bremsstrahlung](#inverse-bremsstrahlung).

### Inverse Bremsstrahlung

$$
e + H^+ + h\nu \rightarrow e + H^+
$$

The inverse process is [Bremsstrahlung](#bremsstrahlung).

---

## Electron-Neutral Interactions

### Elastic Collision

$$
e + H \rightarrow e + H
$$

 

### Collisional Excitation

$$
e + H(n) \rightarrow e + H(m)
$$

with \(m > n\).

The inverse process is [Collisional De-Excitation](#collisional-de-excitation).

### Collisional De-Excitation

$$
e + H(m) \rightarrow e + H(n)
$$

with \(m > n\).

The inverse process is [Collisional Excitation](#collisional-excitation).

### Collisional Ionization

$$
e + H(n) \rightarrow H^+ + e + e
$$

The inverse process is [Three-Body Recombination](#three-body-recombination).

### Radiative Electron Attachment

$$
H + e \rightarrow H^- + h\nu
$$

The inverse process is [Photo-Detachment](#photo-detachment).

### Electron-Impact Detachment

$$
H^- + e \rightarrow H + e + e
$$

The formal inverse process is [Three-Body Electron Attachment](#three-body-electron-attachment).

### Three-Body Electron Attachment

$$
H + e + e \rightarrow H^- + e
$$

The inverse process is [Electron-Impact Detachment](#electron-impact-detachment).

---

## Photon-Atomic Interactions

### Spontaneous Emission

$$
H(m) \rightarrow H(n) + h\nu
$$

with \(m > n\).

The inverse process is [Photo-Excitation](#photo-excitation).

### Photo-Excitation

$$
H(n) + h\nu \rightarrow H(m)
$$

with \(m > n\).

The inverse process is [Spontaneous Emission](#spontaneous-emission).

### Stimulated Emission

$$
H(m) + h\nu \rightarrow H(n) + h\nu + h\nu
$$

with \(m > n\).

The inverse process is [Photo-Excitation](#photo-excitation).

### Photo-Ionization

$$
H(n) + h\nu \rightarrow H^+ + e
$$

The inverse process is [Radiative Recombination](#radiative-recombination).

### Photo-Detachment

$$
H^- + h\nu \rightarrow H + e
$$

The inverse process is [Radiative Electron Attachment](#radiative-electron-attachment).

---

## Ion-Neutral Interactions

### Elastic Collision

$$
H^+ + H \rightarrow H^+ + H
$$

 

### Resonant Charge Exchange

$$
H^+ + H \rightarrow H + H^+
$$

For a pure hydrogenic plasma this reaction is formally symmetric, but it is kinetically important because charge exchange transfers ion and neutral identities between particles with different velocities.

 

### Charge Exchange with Excited Hydrogen

$$
H^+ + H(n) \rightarrow H(n') + H^+
$$

The inverse process is another charge-exchange process with the initial and final internal states interchanged.

### Ion-Impact Excitation

$$
H^+ + H(n) \rightarrow H^+ + H(m)
$$

with \(m > n\).

The inverse process is [Ion-Impact De-Excitation](#ion-impact-de-excitation).

### Ion-Impact De-Excitation

$$
H^+ + H(m) \rightarrow H^+ + H(n)
$$

with \(m > n\).

The inverse process is [Ion-Impact Excitation](#ion-impact-excitation).

### Ion-Impact Ionization

$$
H^+ + H \rightarrow H^+ + H^+ + e
$$

The formal inverse process is [Heavy-Particle Three-Body Recombination](#heavy-particle-three-body-recombination).

### Heavy-Particle Three-Body Recombination

$$
H^+ + H^+ + e \rightarrow H^+ + H
$$

The inverse process is [Ion-Impact Ionization](#ion-impact-ionization).

---

## Neutral-Neutral Interactions

Lower importance compared with electron-impact and ion-neutral processes, but relevant in cold, dense, detached, or high-neutral-density regions.

### Atom-Atom Elastic Collision

$$
H + H \rightarrow H + H
$$

 

### Atom-Molecule Elastic Collision

$$
H + H_2 \rightarrow H + H_2
$$

 

### Molecule-Molecule Elastic Collision

$$
H_2 + H_2 \rightarrow H_2 + H_2
$$

 

### Excitation Transfer

$$
H^* + H \rightarrow H + H^*
$$

This process is self-inverse when the two atoms are the same isotope and only excitation exchange is considered.

### Neutral Quenching

$$
H^* + H \rightarrow H + H
$$

The formal inverse process is [Neutral-Impact Excitation](#neutral-impact-excitation).

### Neutral-Impact Excitation

$$
H + H \rightarrow H^* + H
$$

The inverse process is [Neutral Quenching](#neutral-quenching).

### Penning Ionization

$$
H^* + H^* \rightarrow H^+ + H + e
$$

The formal inverse process is [Three-Body Excited-State Recombination](#three-body-excited-state-recombination).

### Associative Ionization

$$
H^* + H \rightarrow H_2^+ + e
$$

The inverse process is [Dissociative Recombination of Molecular Hydrogen Ion](#dissociative-recombination-of-molecular-hydrogen-ion).

### Three-Body Excited-State Recombination

$$
H^+ + H + e \rightarrow H^* + H^*
$$

The inverse process is [Penning Ionization](#penning-ionization).

### Volume Molecular Formation

$$
H + H + H \rightarrow H_2 + H
$$

The inverse process is [Neutral-Impact Molecular Dissociation](#neutral-impact-molecular-dissociation).

### Neutral-Impact Molecular Dissociation

$$
H_2 + H \rightarrow H + H + H
$$

The inverse process is [Volume Molecular Formation](#volume-molecular-formation).

---

## Ion-Ion Interactions

### Ion-Ion Coulomb Collision

Usually considered in transport or collisional operators.

$$
H^+ + H^+ \rightarrow H^+ + H^+
$$

 

### Mutual Neutralization

$$
H^+ + H^- \rightarrow H + H^*
$$

or,

$$
H^+ + H^- \rightarrow H + H
$$

The inverse process is [Ion-Pair Formation](#ion-pair-formation).

### Ion-Pair Formation

$$
H + H^* \rightarrow H^+ + H^-
$$

or,

$$
H + H \rightarrow H^+ + H^-
$$

The inverse process is [Mutual Neutralization](#mutual-neutralization).

---

## Electron-Molecular Interactions

### Electron-Molecule Elastic Collision

$$
e + H_2 \rightarrow e + H_2
$$

 

### Rotational Excitation

$$
e + H_2(v,J) \rightarrow e + H_2(v,J')
$$

with \(J' > J\).

The inverse process is [Rotational De-Excitation](#rotational-de-excitation).

### Rotational De-Excitation

$$
e + H_2(v,J') \rightarrow e + H_2(v,J)
$$

with \(J' > J\).

The inverse process is [Rotational Excitation](#rotational-excitation).

### Vibrational Excitation

$$
e + H_2(v,J) \rightarrow e + H_2(v',J)
$$

with \(v' > v\).

The inverse process is [Vibrational De-Excitation](#vibrational-de-excitation).

### Vibrational De-Excitation

$$
e + H_2(v',J) \rightarrow e + H_2(v,J)
$$

with \(v' > v\).

The inverse process is [Vibrational Excitation](#vibrational-excitation).

### Molecular Electronic Excitation

$$
e + H_2 \rightarrow e + H_2^*
$$

The inverse process is [Molecular Electronic De-Excitation](#molecular-electronic-de-excitation).

### Molecular Electronic De-Excitation

$$
e + H_2^* \rightarrow e + H_2
$$

The inverse process is [Molecular Electronic Excitation](#molecular-electronic-excitation).

### Electron-Impact Dissociation

$$
H_2 + e \rightarrow H + H + e
$$

The formal inverse process is [Electron-Assisted Three-Body Molecular Formation](#electron-assisted-three-body-molecular-formation).

### Electron-Assisted Three-Body Molecular Formation

$$
H + H + e \rightarrow H_2 + e
$$

The inverse process is [Electron-Impact Dissociation](#electron-impact-dissociation).

### Dissociative Excitation

$$
H_2 + e \rightarrow H + H^* + e
$$

The formal inverse process is [Electron-Assisted Associative De-Excitation](#electron-assisted-associative-de-excitation).

### Electron-Assisted Associative De-Excitation

$$
H + H^* + e \rightarrow H_2 + e
$$

The inverse process is [Dissociative Excitation](#dissociative-excitation).

### Molecular Ionization

$$
H_2 + e \rightarrow H_2^+ + e + e
$$

The formal inverse process is [Three-Body Molecular-Ion Recombination](#three-body-molecular-ion-recombination).

### Three-Body Molecular-Ion Recombination

$$
H_2^+ + e + e \rightarrow H_2 + e
$$

The inverse process is [Molecular Ionization](#molecular-ionization).

### Dissociative Ionization

$$
H_2 + e \rightarrow H^+ + H + e + e
$$

The formal inverse process is [Three-Body Dissociative Recombination](#three-body-dissociative-recombination).

### Three-Body Dissociative Recombination

$$
H^+ + H + e + e \rightarrow H_2 + e
$$

The inverse process is [Dissociative Ionization](#dissociative-ionization).

### Dissociative Attachment

$$
H_2 + e \rightarrow H^- + H
$$

More generally,

$$
H_2(v) + e \rightarrow H^- + H
$$

The inverse process is [Associative Detachment](#associative-detachment).

---

## Electron-Molecular-Ion Interactions

### Dissociation of Molecular Hydrogen Ion

$$
H_2^+ + e \rightarrow H^+ + H + e
$$

The formal inverse process is [Electron-Assisted Formation of Molecular Hydrogen Ion](#electron-assisted-formation-of-molecular-hydrogen-ion).

### Electron-Assisted Formation of Molecular Hydrogen Ion

$$
H^+ + H + e \rightarrow H_2^+ + e
$$

The inverse process is [Dissociation of Molecular Hydrogen Ion](#dissociation-of-molecular-hydrogen-ion).

### Dissociative Ionization of Molecular Hydrogen Ion

$$
H_2^+ + e \rightarrow H^+ + H^+ + e + e
$$

The formal inverse process is [Three-Body Recombination to Molecular Hydrogen Ion](#three-body-recombination-to-molecular-hydrogen-ion).

### Three-Body Recombination to Molecular Hydrogen Ion

$$
H^+ + H^+ + e + e \rightarrow H_2^+ + e
$$

The inverse process is [Dissociative Ionization of Molecular Hydrogen Ion](#dissociative-ionization-of-molecular-hydrogen-ion).

### Dissociative Recombination of Molecular Hydrogen Ion

$$
H_2^+ + e \rightarrow H + H
$$

or,

$$
H_2^+ + e \rightarrow H + H^*
$$

The inverse process is [Associative Ionization](#associative-ionization).

### Dissociative Recombination of Trihydrogen Ion

$$
H_3^+ + e \rightarrow H_2 + H
$$

or,

$$
H_3^+ + e \rightarrow H + H + H
$$

The inverse process is [Associative Ionization to Trihydrogen Ion](#associative-ionization-to-trihydrogen-ion).

### Associative Ionization to Trihydrogen Ion

$$
H_2 + H \rightarrow H_3^+ + e
$$

or,

$$
H + H + H \rightarrow H_3^+ + e
$$

The inverse process is [Dissociative Recombination of Trihydrogen Ion](#dissociative-recombination-of-trihydrogen-ion).

---

## Ion-Molecular Interactions

### Ion-Molecule Elastic Collision

$$
H^+ + H_2 \rightarrow H^+ + H_2
$$

 

### Molecular Charge Exchange

$$
H^+ + H_2 \rightarrow H + H_2^+
$$

The inverse process is [Inverse Molecular Charge Exchange](#inverse-molecular-charge-exchange).

### Inverse Molecular Charge Exchange

$$
H + H_2^+ \rightarrow H^+ + H_2
$$

The inverse process is [Molecular Charge Exchange](#molecular-charge-exchange).

### Dissociative Charge Exchange

$$
H^+ + H_2 \rightarrow H + H^+ + H
$$

The formal inverse process is [Associative Charge Exchange](#associative-charge-exchange).

### Associative Charge Exchange

$$
H + H^+ + H \rightarrow H^+ + H_2
$$

The inverse process is [Dissociative Charge Exchange](#dissociative-charge-exchange).

### Ion-Impact Molecular Dissociation

$$
H^+ + H_2 \rightarrow H^+ + H + H
$$

The formal inverse process is [Ion-Assisted Molecular Formation](#ion-assisted-molecular-formation).

### Ion-Assisted Molecular Formation

$$
H^+ + H + H \rightarrow H^+ + H_2
$$

The inverse process is [Ion-Impact Molecular Dissociation](#ion-impact-molecular-dissociation).

### Ion-Impact Molecular Ionization

$$
H^+ + H_2 \rightarrow H^+ + H_2^+ + e
$$

The formal inverse process is [Heavy-Particle-Assisted Molecular-Ion Recombination](#heavy-particle-assisted-molecular-ion-recombination).

### Heavy-Particle-Assisted Molecular-Ion Recombination

$$
H^+ + H_2^+ + e \rightarrow H^+ + H_2
$$

The inverse process is [Ion-Impact Molecular Ionization](#ion-impact-molecular-ionization).

### Ion Conversion

$$
H_2^+ + H_2 \rightarrow H_3^+ + H
$$

The inverse process is [Reverse Ion Conversion](#reverse-ion-conversion).

### Reverse Ion Conversion

$$
H_3^+ + H \rightarrow H_2^+ + H_2
$$

The inverse process is [Ion Conversion](#ion-conversion).

### Proton Transfer

$$
H_3^+ + H \rightarrow H_2^+ + H_2
$$

The inverse process is [Ion Conversion](#ion-conversion).

---

## Negative-Ion Heavy-Particle Interactions

### Associative Detachment

$$
H^- + H \rightarrow H_2 + e
$$

The inverse process is [Dissociative Attachment](#dissociative-attachment).

### Collisional Detachment by Atoms

$$
H^- + H \rightarrow H + H + e
$$

The formal inverse process is [Collisional Attachment by Atoms](#collisional-attachment-by-atoms).

### Collisional Attachment by Atoms

$$
H + H + e \rightarrow H^- + H
$$

The inverse process is [Collisional Detachment by Atoms](#collisional-detachment-by-atoms).

### Collisional Detachment by Molecules

$$
H^- + H_2 \rightarrow H + H_2 + e
$$

The formal inverse process is [Collisional Attachment by Molecules](#collisional-attachment-by-molecules).

### Collisional Attachment by Molecules

$$
H + H_2 + e \rightarrow H^- + H_2
$$

The inverse process is [Collisional Detachment by Molecules](#collisional-detachment-by-molecules).

### Mutual Neutralization with Molecular Hydrogen Ion

$$
H^- + H_2^+ \rightarrow H_2 + H
$$

The inverse process is [Ion-Pair Formation from Molecules](#ion-pair-formation-from-molecules).

### Mutual Neutralization with Trihydrogen Ion

$$
H^- + H_3^+ \rightarrow H_2 + H_2
$$

The inverse process is [Ion-Pair Formation from Molecular Products](#ion-pair-formation-from-molecular-products).

### Ion-Pair Formation from Molecules

$$
H_2 + H \rightarrow H^- + H_2^+
$$

The inverse process is [Mutual Neutralization with Molecular Hydrogen Ion](#mutual-neutralization-with-molecular-hydrogen-ion).

### Ion-Pair Formation from Molecular Products

$$
H_2 + H_2 \rightarrow H^- + H_3^+
$$

The inverse process is [Mutual Neutralization with Trihydrogen Ion](#mutual-neutralization-with-trihydrogen-ion).

---

## Molecular Assisted Recombination

Molecular assisted recombination, MAR, is not a single elementary reaction. It is a chain of molecular reactions that produces an effective ion sink.

### MAR via Molecular Hydrogen Ion

$$
H^+ + H_2 \rightarrow H + H_2^+
$$

followed by,

$$
H_2^+ + e \rightarrow H + H
$$

Overall,

$$
H^+ + e + H_2 \rightarrow H + H + H
$$

The inverse pathway is associated with [Molecular Charge Exchange](#molecular-charge-exchange), [Inverse Molecular Charge Exchange](#inverse-molecular-charge-exchange), [Dissociative Recombination of Molecular Hydrogen Ion](#dissociative-recombination-of-molecular-hydrogen-ion), and [Associative Ionization](#associative-ionization).

### MAR via Trihydrogen Ion

$$
H_2^+ + H_2 \rightarrow H_3^+ + H
$$

followed by,

$$
H_3^+ + e \rightarrow H_2 + H
$$

or,

$$
H_3^+ + e \rightarrow H + H + H
$$

The inverse pathway is associated with [Ion Conversion](#ion-conversion), [Reverse Ion Conversion](#reverse-ion-conversion), [Dissociative Recombination of Trihydrogen Ion](#dissociative-recombination-of-trihydrogen-ion), and [Associative Ionization to Trihydrogen Ion](#associative-ionization-to-trihydrogen-ion).

### MAR via Negative Ions

$$
H_2 + e \rightarrow H^- + H
$$

followed by,

$$
H^- + H^+ \rightarrow H + H^*
$$

or,

$$
H^- + H^+ \rightarrow H + H
$$

Overall,

$$
H^+ + e + H_2 \rightarrow H + H + H^*
$$

## References and Links:

### See also:

<!-- - Related topic page
- Related code page: `../../code_docs/<page>.md`
- External notes, books, datasets, or wiki pages (use [@key] to cite items in bibliography/bibliography.bib) -->

### Bibliography:

<!-- Use in-text citations with the MkDocs BibTeX syntax where relevant, then render
the page-local references list here: -->

\bibliography

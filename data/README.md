# Data provenance

## `tox21.csv.gz`

The Tox21 data set as distributed by **MoleculeNet**, downloaded from DeepChem's
public bucket:

```
https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/tox21.csv.gz
```

* 7,831 compounds, 12 assay columns plus `mol_id` and `smiles`
* downloaded 24 November 2026; the gzip header records the file as built 21 December 2017
* 123 kB, so it is vendored here rather than downloaded at run time: every
  notebook in this repository runs offline on a laptop

The 12 assays are seven nuclear-receptor (`NR-`) and five stress-response
(`SR-`) panels from the **Tox21 Data Challenge 2014**, run by the NIH/EPA Tox21
consortium. The two used as worked examples in the notebooks are:

| task | what it measures |
|---|---|
| `NR-AR` | androgen receptor agonism (full receptor) |
| `SR-MMP` | disruption of the mitochondrial membrane potential |

### Two things to know before you model it

1. **Blank cells mean "not measured", not "inactive."** Not every compound was
   run in every assay. `mgnn.data` keeps a boolean mask beside the labels and
   `mgnn.train.masked_bce_loss` scores only measured entries. Filling blanks
   with zero would invent thousands of negatives and inflate every metric.
2. **The classes are very unbalanced** — a few percent actives per assay. This
   repository therefore never reports accuracy; it reports ROC-AUC and PR-AUC,
   and offers a positive-class weight in the loss.

### To re-download

```bash
curl -L -o data/tox21.csv.gz \
  https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/tox21.csv.gz
```

### References

* Huang, R. et al. "Tox21Challenge to Build Predictive Models of Nuclear
  Receptor and Stress Response Pathways as Mediated by Exposure to
  Environmental Chemicals and Drugs." *Front. Environ. Sci.* **2016**, *3*, 85.
  DOI: [10.3389/fenvs.2015.00085](https://doi.org/10.3389/fenvs.2015.00085)
* Wu, Z. et al. "MoleculeNet: A Benchmark for Molecular Machine Learning."
  *Chem. Sci.* **2018**, *9*, 513-530.
  DOI: [10.1039/C7SC02664A](https://doi.org/10.1039/C7SC02664A)

### Licence note

The Tox21 data is public, produced by US federal agencies (NIH/NCATS, EPA, FDA)
and redistributed by MoleculeNet/DeepChem (MIT licence). It is included here for
teaching. The MIT licence in this repository covers the **code**, not the data.

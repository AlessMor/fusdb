from fusdb.registry import RELATIONS


def test_volume_integrated_radiation_relations_expose_geometry_measure():
    names = (
        "Bremsstrahlung radiation",
        "Hydrogenic bremsstrahlung (cfspopcon)",
        "Impurity bremsstrahlung from total and hydrogenic",
        "Synchrotron radiation",
        "Impurity line radiation (Mavrin coronal)",
        "Impurity line radiation (Post-Jensen)",
        "Impurity line radiation (Mavrin noncoronal)",
        "Impurity line radiation (PROCESS coronal tables)",
        "Species-sum radiated power (PROCESS coronal tables)",
        "Impurity line radiation (radas coronal)",
    )

    for name in names:
        relation = RELATIONS.get(name)
        assert "w_V" in relation.constant_names, name

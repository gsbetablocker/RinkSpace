from rinkspace.session.material_tracks import qualify_reference_links


def edge(a, b, source):
    return dict(a=a, b=b, pairs=[[i, i+100] for i in source], unambiguous=False)


def test_disjoint_visible_spans_of_one_reference_keep_the_same_identity():
    links = [edge(0, 2, range(6)), edge(0, 3, range(10, 16))]
    qualify_reference_links(links, ['B1'])
    assert all(e['identity_unambiguous'] for e in links)


def test_same_reference_support_claiming_two_branches_remains_pending():
    links = [edge(0, 2, range(6)), edge(0, 3, range(1, 7))]
    qualify_reference_links(links, ['B1'])
    assert not any(e['identity_unambiguous'] for e in links)


def test_different_physical_identities_claiming_one_fragment_stay_pending():
    links = [edge(0, 2, range(6)), edge(1, 2, range(10, 16))]
    qualify_reference_links(links, ['B1', 'B2'])
    assert not any(e['identity_unambiguous'] for e in links)


def test_two_reference_spans_with_same_identity_do_not_create_a_false_conflict():
    links = [edge(0, 2, range(6)), edge(1, 2, range(10, 16))]
    qualify_reference_links(links, ['B1', 'B1'])
    assert all(e['identity_unambiguous'] for e in links)

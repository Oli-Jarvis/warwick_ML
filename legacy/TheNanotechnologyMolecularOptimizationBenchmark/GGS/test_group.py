import pytest
from .group import Group

def test_occupy_attachment_point():
    g = Group('pyrazole', 'N1C=CC=N1', all_attachment=True)
    assert g.name == 'pyrazole'
    g.occupy_attachment_point(0)
    assert g.occupied_attachment_points == [0]
    ref_attachment_point_occupation = [1, 0, 0, 0]
    assert all(i == j for i, j in zip(g.attachment_point_occupation, ref_attachment_point_occupation))
    assert g.occupy_attachment_point(0) == False

def test_next_avail_attachment():
    g = Group('toluene', 'C-C1=C(*1)-C(*1)=C(*1)-C(*1)=C-1*1')
    print(g.attachment_points)
    assert g.name == 'toluene'
    assert len(g.attachment_points) == 5
    assert g.occupied_attachment_points == []
    assert g.next_avail_attachment(0) == 0
    assert g.occupy_attachment_point(3) == True
    ref_attachment_point_occupation = [0, 0, 0, 1, 0, 0]
    assert all(i == j for i, j in zip(g.attachment_point_occupation, ref_attachment_point_occupation))
    assert g.next_avail_attachment(4) == 2
    #assert g.occupy_attachment_point(0) == True
    #assert g.occupied_attachment_points == [0]
    #ref_attachment_point_occupation = [1, 0, 0, 0, 0, 0]
    #assert all(i == j for i, j in zip(g.attachment_point_occupation, ref_attachment_point_occupation))

    """
    for i in range(0, 2*len(g.attachment_points)):
        assert g.next_avail_attachment(0) == 1
    
    assert g.next_avail_attachment(5) == 5
    assert g.next_avail_attachment(6) == 1
    assert g.occupy_attachment_point(1) == True
    assert g.next_avail_attachment(6) == 2

    ref_attachment_point_occupation = [1, 1, 0, 0, 0, 0]
    assert all(i == j for i, j in zip(g.attachment_point_occupation, ref_attachment_point_occupation))
    assert len(g.occupied_attachment_points) == 2
    ref_occupied_attachment_points = [0, 1]
    assert all(i == j for i, j in zip(g.occupied_attachment_points, ref_occupied_attachment_points))
    """


    
def test_connect_to_last_attachment_possible():        
    g = Group('trifluoromethane', 'F - C(-F)(-F) * 1', all_attachment=True)
    assert g.connect_to_last_attachment_possible() == True
    assert g.occupy_attachment_point(0) == True
    assert g.connect_to_last_attachment_possible() == False
    
    assert g.next_avail_attachment(42) == -1
    
def test_free_attachment_point():
    g = Group('toluene', 'CC1=CC=CC=C1', all_attachment=True)
    with pytest.raises(ValueError):
        g.occupy_attachment_point(42)
    g.occupy_attachment_point(0)
    assert g.attachment_point_start == 0
    assert g.free_attachment_point(0) == False
    assert g.occupy_attachment_point(1) == True
    assert g.occupied_attachment_points == [0, 1]
    assert g.free_attachment_point(0) == False
    assert g.free_attachment_point(1) == True
    assert g.occupied_attachment_points == [0]
    assert g.attachment_point_occupation[1] == 0
    assert g.attachment_point_occupation[0] == 1
    
def test_free_last_attachment_point():
    g = Group('toluene', 'CC1=CC=CC=C1', all_attachment=True)
    assert g.free_last_attachment_point() == False
    g.occupy_attachment_point(0)
    assert g.free_last_attachment_point() == False
    g.occupy_attachment_point(1)
    assert g.free_last_attachment_point() == True

def test_get_free_attachment_points():
    g = Group('toluene', 'CC1=CC=CC=C1', all_attachment=True)
    assert g.get_free_attachment_points() == [0, 1, 2, 3, 4, 5]
    g.occupy_attachment_point(0)
    assert g.get_free_attachment_points() == [1, 2, 3, 4, 5]
    g.occupy_attachment_point(2)
    assert g.get_free_attachment_points() == [1, 3, 4, 5]
def test_reserve_attachment_point():
    g = Group('toluene', 'CC1=CC=CC=C1', all_attachment=True)
    with pytest.raises(ValueError):
        g.reserve_attachment_point(42)
    with pytest.raises(ValueError):
        g.reserve_attachment_point(0)

    assert g.reserved_attachment_point == -1

    assert g.occupy_attachment_point(0) == True
    assert g.reserve_attachment_point(1) == True
    assert g.reserved_attachment_point == 1
    assert g.reserve_attachment_point(2) == True
    assert g.reserved_attachment_point == 2
    assert g.occupy_attachment_point(2) == True
    assert g.reserved_attachment_point == -1
    #assert g.attachment_point_start == 0

    
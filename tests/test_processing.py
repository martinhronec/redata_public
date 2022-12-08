import redata
import redata.processing

def test_get_general_prague_part():
    assert redata.processing.get_general_prague_part('Praha 10 - Vršovice') == 'Praha 10'
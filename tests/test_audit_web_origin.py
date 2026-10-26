"""The local workbench must not trust arbitrary DNS Host names."""
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from test_workbench import http_workbench, project


@pytest.mark.parametrize("method,path", [("GET", "/api/state"), ("POST", "/api/project")])
def test_foreign_host_is_rejected_even_when_origin_matches_it(http_workbench, method, path):
    base, app_root = http_workbench
    request = Request(base + path,
                      data=b'{"action":"save"}' if method == "POST" else None,
                      method=method,
                      headers={"Host": "untrusted.example", "Origin": "http://untrusted.example",
                               "Content-Type": "application/json"})
    with pytest.raises(HTTPError) as error:
        urlopen(request, timeout=5)
    assert error.value.code == 400

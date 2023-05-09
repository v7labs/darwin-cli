import pytest

from darwin.future.core.client import Client, DarwinConfig
from darwin.future.data_objects.team import Team, TeamMember


@pytest.fixture
def base_config() -> DarwinConfig:
    return DarwinConfig(
        api_key="test_key",
        base_url="http://test_url.com/",
        api_endpoint="http://test_url.com/api/",
        default_team="default-team",
        teams={},
    )


@pytest.fixture
def base_client(base_config: DarwinConfig) -> Client:
    return Client(base_config)


@pytest.fixture
def base_team_json() -> dict:
    return {"slug": "test-team", "id": "0"}


@pytest.fixture
def base_team(base_team_json: dict) -> Team:
    return Team.parse_obj(base_team_json)


@pytest.fixture
def base_team_member_json() -> dict:
    return {
        "email": "email",
        "id": "0",
        "first_name": "first",
        "last_name": "last",
        "role": "member",
        "team_id": "0",
        "user_id": "0",
    }


@pytest.fixture
def base_team_member(base_team_member_json: dict) -> TeamMember:
    return TeamMember.parse_obj(base_team_member_json)

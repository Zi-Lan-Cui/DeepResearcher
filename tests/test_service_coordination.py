from deepsearch_agent.service.coordination import (
    DATABASE_MIGRATION_LOCK_ID,
    POSTGRES_ADVISORY_LOCK_IDS,
    RUN_CLAIM_CAPACITY_LOCK_ID,
    WORKER_STARTUP_RECOVERY_LOCK_ID,
)


def test_postgres_advisory_lock_ids_are_stable_and_unique():
    assert POSTGRES_ADVISORY_LOCK_IDS == (
        DATABASE_MIGRATION_LOCK_ID,
        RUN_CLAIM_CAPACITY_LOCK_ID,
        WORKER_STARTUP_RECOVERY_LOCK_ID,
    )
    assert POSTGRES_ADVISORY_LOCK_IDS == (731_904_620, 731_904_621, 731_904_622)
    assert len(set(POSTGRES_ADVISORY_LOCK_IDS)) == len(POSTGRES_ADVISORY_LOCK_IDS)

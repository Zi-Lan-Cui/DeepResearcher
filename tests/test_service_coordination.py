from deepresearcher.service.coordination import (
    DATABASE_MIGRATION_LOCK_ID,
    POSTGRES_ADVISORY_LOCK_IDS,
    RETIRED_RUN_CLAIM_CAPACITY_LOCK_ID,
    RUN_ADMISSION_LOCK_ID,
    RUN_RETENTION_SWEEP_LOCK_ID,
    WORKER_STARTUP_RECOVERY_LOCK_ID,
)


def test_postgres_advisory_lock_ids_are_stable_and_unique():
    assert POSTGRES_ADVISORY_LOCK_IDS == (
        DATABASE_MIGRATION_LOCK_ID,
        WORKER_STARTUP_RECOVERY_LOCK_ID,
        RUN_ADMISSION_LOCK_ID,
        RUN_RETENTION_SWEEP_LOCK_ID,
    )
    assert POSTGRES_ADVISORY_LOCK_IDS == (
        731_904_620,
        731_904_622,
        731_904_623,
        731_904_624,
    )
    assert len(set(POSTGRES_ADVISORY_LOCK_IDS)) == len(POSTGRES_ADVISORY_LOCK_IDS)
    # 退役编号不得与在用锁重合,否则旧进程与新进程会在同一把锁上互相阻塞。
    assert RETIRED_RUN_CLAIM_CAPACITY_LOCK_ID not in POSTGRES_ADVISORY_LOCK_IDS

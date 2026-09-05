"""Stable PostgreSQL coordination keys shared by independent service processes.

Advisory lock keys are a database-wide coordination contract.  Keep them named,
unique, and stable across rolling deployments: changing a key while old processes
are still alive would split one critical section into two unrelated locks.
"""

DATABASE_MIGRATION_LOCK_ID = 731_904_620
RUN_CLAIM_CAPACITY_LOCK_ID = 731_904_621
WORKER_STARTUP_RECOVERY_LOCK_ID = 731_904_622

POSTGRES_ADVISORY_LOCK_IDS = (
    DATABASE_MIGRATION_LOCK_ID,
    RUN_CLAIM_CAPACITY_LOCK_ID,
    WORKER_STARTUP_RECOVERY_LOCK_ID,
)

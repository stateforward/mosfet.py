export type ScheduleHandle = unknown;

export type Scheduler = {
  schedule: (flush: () => void) => ScheduleHandle;
  cancel: (handle: ScheduleHandle) => void;
};

const ZERO_DELAY_MS = 0;

/**
 * DOM timeout scheduler used to coalesce pointer samples.
 *
 * Inputs: optional `schedule` and `cancel` adapters. Default schedule is
 * `setTimeout` with `ZERO_DELAY_MS`.
 * Outputs: a `Scheduler` whose `schedule(flush)` queues `flush` once.
 * Ownership: caller owns the returned scheduler and every handle.
 * Lifetime: until `cancel` or page teardown.
 * Concurrency: each handle is independent; `coalesceLatest` uses one handle.
 * Failure modes: a missing global timer throws from the default adapter.
 * Units: delay is milliseconds.
 * Classification: external-system (clock).
 */
export function timeoutScheduler(args: {
  schedule?: (fields: { flush: () => void; delay: number }) => ScheduleHandle;
  cancel?: (handle: ScheduleHandle) => void;
} = {}): Scheduler {
  const schedule = args.schedule ?? ((fields: { flush: () => void; delay: number }) => globalThis.setTimeout(fields.flush, fields.delay));
  const cancel = args.cancel ?? ((handle) => {
    globalThis.clearTimeout(handle as ReturnType<typeof globalThis.setTimeout>);
  });
  return {
    schedule: (flush) => schedule({ flush, delay: ZERO_DELAY_MS }),
    cancel,
  };
}

/**
 * Keep only the latest pushed value and emit it on the next scheduler flush.
 *
 * Inputs: `scheduler` and `emit`.
 * Outputs: `push` and `dispose`. `push` replaces the pending value; `dispose`
 * cancels a pending handle and drops the value.
 * Ownership: caller owns the coalescer and must `dispose`.
 * Lifetime: until `dispose`.
 * Concurrency: one pending handle; overlapping `push` is last-write-wins.
 * Failure modes: `emit` exceptions propagate from the scheduler flush.
 * Units: none.
 * Classification: external-system (clock).
 */
export function coalesceLatest<T>(args: {
  scheduler: Scheduler;
  emit: (value: T) => void;
}): { push: (value: T) => void; dispose: () => void } {
  let latest: T | null = null;
  let handle: ScheduleHandle | 0 = 0;
  const flush = (): void => {
    handle = 0;
    const value = latest;
    latest = null;
    if (value !== null) args.emit(value);
  };
  return {
    push(value: T): void {
      latest = value;
      if (handle !== 0) return;
      handle = args.scheduler.schedule(flush);
    },
    dispose(): void {
      if (handle !== 0) args.scheduler.cancel(handle);
      handle = 0;
      latest = null;
    },
  };
}

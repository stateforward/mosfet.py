export type ScheduleHandle = unknown;

export type Scheduler = {
  schedule: (flush: () => void) => ScheduleHandle;
  cancel: (handle: ScheduleHandle) => void;
};

export function timeoutScheduler(args: {
  schedule?: (flush: () => void, delay: number) => ScheduleHandle;
  cancel?: (handle: ScheduleHandle) => void;
} = {}): Scheduler {
  const schedule = args.schedule ?? ((flush, delay) => globalThis.setTimeout(flush, delay));
  const cancel = args.cancel ?? ((handle) => {
    globalThis.clearTimeout(handle as ReturnType<typeof globalThis.setTimeout>);
  });
  return {
    schedule: (flush) => schedule(flush, 0),
    cancel,
  };
}

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

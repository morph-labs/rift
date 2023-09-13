export class Store<T> {
  value: T;
  listeners: ((state: T) => void)[];

  constructor(initialState: T, listeners?: ((state: T) => void)[]) {
    this.value = initialState;
    this.listeners = listeners ?? [];
  }

  set(newState: T) {
    this.value = newState;
    this.notifyListeners();
  }

  update(updater: (prevState: T) => T) {
    this.value = updater(this.value);
    // console.log('new state:', this.value)
    this.notifyListeners();
  }

  subscribe(listener: (state: T) => void) {
    this.listeners.push(listener);
  }

  notifyListeners() {
    for (const listener of this.listeners) {
      listener(this.value);
    }
  }
}

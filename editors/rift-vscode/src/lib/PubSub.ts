const registry: { [key: string]: (...args: any) => void | null } = {};

export const pub = (key: string, ...args: any) => {
  console.log("pubbing");
  console.log(key);
  //   if (!registry[key]) return;
  const fn = registry[key];
  if (!fn) throw new Error("published to an unawaited key");
  fn(...args);
};

export const sub = (key: string, fn: (...args: any) => void) => {
  console.log("subbing");
  console.log(key);
  if (registry[key]) {
    delete registry[key];
  }
  registry[key] = fn;
};

const PubSub = { pub, sub };

export default PubSub;

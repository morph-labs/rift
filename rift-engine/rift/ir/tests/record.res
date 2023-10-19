type r = {x?: option<int>, y: option<string>}

let useInConditional = x => {
  if x {
    useState()
  }
}

let useAtToplevel = x => {
  useState([1,2,[3,4]], ("a", "b"))
}

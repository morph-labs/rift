// @ts-ignore 

// This script will be run within the webview itself
// It cannot access the main VS Code APIs directly.
(function () {
  // const originalConsoleLog = console.log;
  // console.log = (...args) => {
  //   originalConsoleLog.apply(console, args.map(arg => 
  //     typeof arg === 'object' ? JSON.parse(JSON.stringify(arg)) : arg
  //   ));
  // }; 

  function copySvg(color = 'var(--vscode-panelTitle-inactiveForeground)') {
    const svgNS = "http://www.w3.org/2000/svg";
  
    // Create the <svg> element
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('fill', color);
    svg.setAttribute('height', '20');
    svg.setAttribute('width', '20');
    svg.setAttribute('version', '1.1');
    svg.setAttribute('id', 'Capa_1');
    svg.setAttribute('viewBox', '0 0 352.804 352.804');
    svg.setAttribute('xml:space', 'preserve');

    // Create the <g> element
    const g = document.createElementNS(svgNS, 'g');
    svg.appendChild(g);

    // Create the <path> element
    const path = document.createElementNS(svgNS, 'path');
    path.setAttribute('d', 'M318.54,57.282h-47.652V15c0-8.284-6.716-15-15-15H34.264c-8.284,0-15,6.716-15,15v265.522c0,8.284,6.716,15,15,15h47.651 v42.281c0,8.284,6.716,15,15,15H318.54c8.284,0,15-6.716,15-15V72.282C333.54,63.998,326.824,57.282,318.54,57.282z M49.264,265.522V30h191.623v27.282H96.916c-8.284,0-15,6.716-15,15v193.24H49.264z M303.54,322.804H111.916V87.282H303.54V322.804 z');
    g.appendChild(path);

    return svg;
}



  function resetSvg() {
    let xmlns = "http://www.w3.org/2000/svg";

    // Create main SVG element
    let svgElem = document.createElementNS(xmlns, "svg");
    svgElem.setAttributeNS(null, "width", "24");
    svgElem.setAttributeNS(null, "height", "24");
    svgElem.setAttributeNS(null, "viewBox", "0 0 25 25");
    svgElem.setAttributeNS(null, "fill", "none");

    // Create path element
    let pathElem = document.createElementNS(xmlns, "path");
    pathElem.setAttributeNS(null, "d", "M4.56189 13.5L4.14285 13.9294L4.5724 14.3486L4.99144 13.9189L4.56189 13.5ZM9.92427 15.9243L15.9243 9.92427L15.0757 9.07574L9.07574 15.0757L9.92427 15.9243ZM9.07574 9.92426L15.0757 15.9243L15.9243 15.0757L9.92426 9.07574L9.07574 9.92426ZM19.9 12.5C19.9 16.5869 16.5869 19.9 12.5 19.9V21.1C17.2496 21.1 21.1 17.2496 21.1 12.5H19.9ZM5.1 12.5C5.1 8.41309 8.41309 5.1 12.5 5.1V3.9C7.75035 3.9 3.9 7.75035 3.9 12.5H5.1ZM12.5 5.1C16.5869 5.1 19.9 8.41309 19.9 12.5H21.1C21.1 7.75035 17.2496 3.9 12.5 3.9V5.1ZM5.15728 13.4258C5.1195 13.1227 5.1 12.8138 5.1 12.5H3.9C3.9 12.8635 3.92259 13.2221 3.9665 13.5742L5.15728 13.4258ZM12.5 19.9C9.9571 19.9 7.71347 18.6179 6.38048 16.6621L5.38888 17.3379C6.93584 19.6076 9.54355 21.1 12.5 21.1V19.9ZM4.99144 13.9189L7.42955 11.4189L6.57045 10.5811L4.13235 13.0811L4.99144 13.9189ZM4.98094 13.0706L2.41905 10.5706L1.58095 11.4294L4.14285 13.9294L4.98094 13.0706Z");
    pathElem.setAttributeNS(null, "fill", "var(--vscode-icon-foreground)");

    // Append path to SVG
    svgElem.appendChild(pathElem);

    return svgElem
}




  const riftSvg = (color = 'var(--vscode-panelTitle-inactiveForeground)') => {
    let svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '24');
    svg.setAttribute('height', '24');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('fill', 'none');
    let path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('fill-rule', 'evenodd');
    path.setAttribute('clip-rule', 'evenodd');
    path.setAttribute('d', 'M4.14645 4.04907C4.14645 4.07615 3.2135 7.54672 2.07322 11.7616C0.93295 15.9765 -1.71e-06 19.4761 -1.71e-06 19.5386C-1.71e-06 19.6127 0.877477 19.6522 2.52149 19.6522C3.90831 19.6522 5.04298 19.6261 5.04298 19.5941C5.04298 19.5621 5.97593 16.0875 7.11621 11.8726C8.25648 7.65772 9.18943 4.16218 9.18943 4.10457C9.18943 4.04218 8.17131 4 6.66794 4C5.28112 4 4.14645 4.02209 4.14645 4.04907ZM11.5428 4.04451C11.5428 4.06905 11.316 4.9308 11.0387 5.95941C10.7615 6.98812 10.5346 7.86731 10.5345 7.91304C10.5343 7.95878 11.667 7.9963 13.0513 7.9963H15.5686L15.693 7.52451C15.7614 7.26498 15.9903 6.41443 16.2016 5.63426C16.4129 4.8541 16.5858 4.16729 16.5858 4.1079C16.5858 4.04118 15.6223 4 14.0643 4C12.6775 4 11.5428 4.02009 11.5428 4.04451ZM18.8367 4.41628C18.3043 6.40888 16.7977 11.9668 16.7464 12.1269C16.6845 12.32 16.8053 12.3284 19.2363 12.3L21.7911 12.2701L22.8272 8.44033C23.3971 6.33395 23.8941 4.47323 23.9316 4.30527L24 4H21.4739H18.9478L18.8367 4.41628Z');
    
    // let fillColor = getComputedStyle(document.documentElement).getPropertyValue('--vscode-panelTitle-inactiveForeground');
    path.setAttribute('fill', color)
    svg.appendChild(path);
    svg.classList.add('mr-2', 'min-w-[24px]')
    return svg
  }

  const userSvg = () => {
    let svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '24');
    svg.setAttribute('height', '24');
    svg.setAttribute('viewBox', '0 0 20 20');

    let title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    title.textContent = 'profile_round [#1342]';
    svg.appendChild(title);

    let desc = document.createElementNS('http://www.w3.org/2000/svg', 'desc');
    desc.textContent = 'Created with Sketch.';
    svg.appendChild(desc);

    let gOuter = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    gOuter.setAttribute('stroke', 'none');
    gOuter.setAttribute('stroke-width', '1');
    gOuter.setAttribute('fill', 'none');
    gOuter.setAttribute('fill-rule', 'evenodd');

    let gMiddle = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    gMiddle.setAttribute('transform', 'translate(-140.000000, -2159.000000)');
    // let fillColor = getComputedStyle(document.documentElement).getPropertyValue('--vscode-panelTitle-inactiveForeground');
    gMiddle.setAttribute('fill', 'var(--vscode-panelTitle-inactiveForeground)'); // matching the color scheme of the previous SVG

    let gInner = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    gInner.setAttribute('transform', 'translate(56.000000, 160.000000)');

    let path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', 'M100.562548,2016.99998 L87.4381713,2016.99998 C86.7317804,2016.99998 86.2101535,2016.30298 86.4765813,2015.66198 C87.7127655,2012.69798 90.6169306,2010.99998 93.9998492,2010.99998 C97.3837885,2010.99998 100.287954,2012.69798 101.524138,2015.66198 C101.790566,2016.30298 101.268939,2016.99998 100.562548,2016.99998 M89.9166645,2004.99998 C89.9166645,2002.79398 91.7489936,2000.99998 93.9998492,2000.99998 C96.2517256,2000.99998 98.0830339,2002.79398 98.0830339,2004.99998 C98.0830339,2007.20598 96.2517256,2008.99998 93.9998492,2008.99998 C91.7489936,2008.99998 89.9166645,2007.20598 89.9166645,2004.99998 M103.955674,2016.63598 C103.213556,2013.27698 100.892265,2010.79798 97.837022,2009.67298 C99.4560048,2008.39598 100.400241,2006.33098 100.053171,2004.06998 C99.6509769,2001.44698 97.4235996,1999.34798 94.7348224,1999.04198 C91.0232075,1998.61898 87.8750721,2001.44898 87.8750721,2004.99998 C87.8750721,2006.88998 88.7692896,2008.57398 90.1636971,2009.67298 C87.1074334,2010.79798 84.7871636,2013.27698 84.044024,2016.63598 C83.7745338,2017.85698 84.7789973,2018.99998 86.0539717,2018.99998 L101.945727,2018.99998 C103.221722,2018.99998 104.226185,2017.85698 103.955674,2016.63598');

    gInner.appendChild(path);
    gMiddle.appendChild(gInner);
    gOuter.appendChild(gMiddle);
    svg.appendChild(gOuter);

    svg.classList.add('mr-2')
    return svg
  }

  const microlightReset = () => {
    document.querySelectorAll('code').forEach(node => node.classList.add('code'))
    document.querySelectorAll('pre').forEach(preblock => {
      preblock.classList.add(
      "p-2",
      "my-2",
      "block",
      "overflow-x-scroll"
    )
    preblock.querySelectorAll('#copy').forEach(copy => copy.parentElement.removeChild(copy))
    const copyContent = preblock.querySelector('.code').innerText
    const copyButton = document.createElement('button')
    copyButton.id = 'copy'
    copyButton.className = 'flex text-sm py-1 mb-1 text-[var(--vscode-panelTitle-inactiveForeground)]'
    copyButton.appendChild(copySvg())
    const copyCodeWords = document.createElement('p')
    copyCodeWords.innerText = ' copy'
    copyButton.appendChild(copyCodeWords)
    copyButton.addEventListener('click', () => {
      // navigator.clipboard.writeText(copyContent)
      vscode.postMessage({command: 'copyText', content: copyContent})
    })
    preblock.insertBefore(copyButton, preblock.firstChild)
    microlight.reset('code')
  })
  }

  function fixCodeBlocks(response) {
    // Use a regular expression to find all occurrences of the substring in the string
    const REGEX_CODEBLOCK = new RegExp('\`\`\`', 'g');
    const matches = response.match(REGEX_CODEBLOCK);
  
    // Return the number of occurrences of the substring in the response, check if even
    const count = matches ? matches.length : 0;
    if (count % 2 === 0) {
      return response;
    } else {
      // else append ``` to the end to make the last code block complete
      return response.concat('\n\`\`\`');
    }
  }

  var converter = new showdown.Converter({
    omitExtraWLInCodeBlocks: true, 
    simplifiedAutoLink: true,
    excludeTrailingPunctuationFromURLs: true,
    literalMidWordUnderscores: true,
    simpleLineBreaks: true
  });

  function textToFormattedHTML(text) {

    // console.log('textToFormattedHTML. text is, ')
    // console.log(text)
    console.log('text conversion')
    text = converter.makeHtml(fixCodeBlocks(text))
    // console.log('formattedHTML is...')
    // console.log(text)
    return text
  }

  microlightReset()
  const consoleLog = (...args) => {
    console.log(args.map(arg =>
      typeof arg === 'object' ? JSON.parse(JSON.stringify(arg)) : arg
    ))
  }

  const vscode = acquireVsCodeApi();

  console.dir(vscode)

  const DEFAULT_STATE = {
    history: [],
  };
    state = vscode.getState();
  if (!state) {
    vscode.setState(DEFAULT_STATE);
  }

   const resetState = () => {
          vscode.setState({history: []});
          renderMessagesFromState(vscode.getState());
   }

  const genChatElement = (isUserInput, dataId = vscode.getState().history.length) => {
    if(!isUserInput) { // if is response
    const div = document.createElement('div')
    div.className = 'w-full text-md p-2 focus:outline-none min-h-8 bg-[var(--vscode-panel-background)] flex hidden flex-row'
    const textbox = document.createElement('div')
    textbox.id = `response`
    textbox.setAttribute('data-id', dataId)
    textbox.className = 'w-full text-md min-h-8'
    div.hasSvg = false
    div.svgOn = () => {
      if(!div.hasSvg) {
        div.classList.remove('hidden')
        div.insertBefore(riftSvg(), div.firstChild)
        div.hasSvg = true
      }
    }
    // div.svgOff = () => { might nuke..dont think I need
    //   if(div.hasSvg){
    //     div.removeChild(div.querySelector('svg'))
    //     div.hasSvg = false
    //   }
    // }
    if(dataId !== 'new') {div.svgOn()}
    div.appendChild(textbox)
    console.log('generating Element with data-id attribute: ', dataId)
    return div
    } 
    else if (isUserInput) {
      const div = document.createElement('div')
      div.className = 'w-full text-md p-2 min-h-8 bg-[var(--vscode-input-background)] flex flex-row'
      const textbox = document.createElement('textarea')
      textbox.className = 'w-full min-h-8 block outline-none focus:outline-none bg-transparent resize-none'
      
      textbox.autoResize = () => {
        textbox.style.height = 'auto';
        textbox.style.height = textbox.scrollHeight + 'px';
      }

      function autoResize() {
        this.style.height = 'auto';
        this.style.height = this.scrollHeight + 'px';
      }
      textbox.addEventListener('input', () => textbox.autoResize(), false);
      textbox.setAttribute('data-id', dataId)
      console.log('generating Element with data-id attribute: ', dataId)
      textbox.placeholder = 'Ask questions and get answers about the current code window.'
      textbox.addEventListener('blur', function (e) {
        renderMessagesFromState(vscode.getState(), e.target.value)
      })
      textbox.addEventListener('keydown', function (e) {
        if (e.key === "Enter") { // 13 is the Enter key code
          e.preventDefault();  // Prevent default Enter key action
          if(e.shiftKey) {
            this.value = this.value + '\n'
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
            return 
          }
          this.style.height = 'auto';
          this.style.height = this.scrollHeight + 'px';
          console.log('You pressed Enter on the div!');
          this.blur()
          document.querySelectorAll('div#response')[document.querySelectorAll('div#response').length-1].innerHTML = '...'

          if (this.getAttribute('data-id') !== 'new') { // if user clicked on earlier chat input
            console.log('data-id not new. data-id is')
            console.log(this.getAttribute('data-id'))
            let num = Number(this.getAttribute('data-id'))
            vscode.getState().history[num] = { role: "user", content: this.value }

            removeStatePastId(num)
            vscode.getState().history.push({ role: 'user', content: this.value })
            renderMessagesFromState(vscode.getState())
          } else {
            vscode.getState().history.push({ role: 'user', content: this.value })
            renderMessagesFromState(vscode.getState())
          }

          vscode.postMessage({
            type: 'message',
              messages: vscode.getState().history.slice(0, -1), // don't want to include what we just pushed :()
            message: this.value
          })

        }
      });
      div.appendChild(userSvg())
      div.appendChild(textbox)
      return div
    }
  }

  const HEADER_HEIGHT = 35
  const genHeader = () => {
    const header = document.createElement('div')
    header.className = `fixed top-0 w-full h-[${HEADER_HEIGHT}px] p-2 flex justify-between bg-[var(--vscode-panel-background)] z-10`
    const button = document.createElement('button')
    const name = document.createElement('div')
    name.className = 'flex flex-row text-xl items-center text-[var(--vscode-icon-foreground)]'
    name.appendChild(riftSvg('var(--vscode-icon-foreground)'))
    name.appendChild(document.createTextNode('Rift'))
    header.appendChild(name)
    header.appendChild(button)
    // button.className = 'bg-[var(--vscode-panelTitle-inactiveForeground)] p-1 rounded-full text-[var(--vscode-panel-background)]'
    button.className = 'flex items-center justify-center'
    button.appendChild(resetSvg())
      button.addEventListener('click', () => {
          resetState();
      });
    return header
  }


  console.log('running chat.js')
  document.body.classList.add('p-0', `pt-[${HEADER_HEIGHT}px]`)
  function removeStatePastId(id) { // IF there's state to remove
    id = Number(id)
      if (id > vscode.getState().history.length) throw new Error()
      if (id === vscode.getState().history.length) return
    console.log('remove state past id(' + id + ') called')
    console.log('state is')
    consoleLog(vscode.getState().history)
    console.log(`state.history.slice(0, ${id})`)
    consoleLog(vscode.getState().history.slice(0, id))
    const new_history = vscode.getState().history.slice(0, id)
    vscode.setState({history: new_history})
    console.log('new state: ')
    consoleLog(vscode.getState().history)
  }


  function renderMessagesFromState(state, currentText = '') {
    console.log('render called')
    
    function removeAllNodes() {
      while (document.body.firstChild) {
        document.body.removeChild(document.body.firstChild);
      }
    }
    const addNewInputAndResponseDiv = () => {

      const userInput = genChatElement(true, 'new')
      
      //Repaste the test after losing focus if necessary
      if (currentText.length > 0) {
        textArea = userInput.querySelector('textarea')
        if (textArea) {
          textArea.value = currentText
        }
      }
    
      document.body.appendChild(userInput)
      userInput.focus()
      const responseDiv = genChatElement(false, 'new')
      document.body.appendChild(responseDiv)
    }

      // vscode.setState(vscode.getState())

      removeAllNodes()
    console.log('render messages from state called. state:')
      console.log(vscode.getState())
      
    document.body.appendChild(genHeader())
      for (let i = 0; i < vscode.getState().history.length; i++) {
          const message = vscode.getState().history[i]
      if (message.role === 'user') {
        const userInputContainer = genChatElement(true, i)
        const userInput = userInputContainer.querySelector('textarea')
        userInput.value = message.content
        userInputContainer.classList.add('bg-[var(--vscode-input-background)]')
        // userInput.classList.add('bg-[var(--vscode-input-background)]')
        document.body.appendChild(userInputContainer)
        userInput.autoResize()
      }
      else if (message.role === 'assistant') {
        const div = genChatElement(false, i)
        div.querySelector('#response').innerHTML = textToFormattedHTML(message.content)
        // div.innerHTML = workingText
        document.body.appendChild(div)
      } else throw new Error('message.role was not assistant or user')
    }

      if (!vscode.getState().history.length || vscode.getState().history[vscode.getState().history.length - 1].role === 'assistant') addNewInputAndResponseDiv()
    else document.body.appendChild(genChatElement(false, 'new'))

    console.log('rendered divs from state: ')
    console.log(document.querySelectorAll('[data-id]'))
    microlightReset()
  }

    renderMessagesFromState(vscode.getState())

  
  // Handle messages sent from the extension to the webview
  window.addEventListener("message", (event) => {
    if (event.data.type === 'progress') {
      //event.data.data as ChatHelperProgress
      const isDone = event.data.data.done
      const responseDiv = document.querySelectorAll('div#response')[document.querySelectorAll('div#response').length-1]
      if (!responseDiv) {throw new Error()}
      if(event.data.data.response === '') {
        return} // the lsp sends a blank message right away -- ignore so the loading shows up.
        responseDiv.innerHTML = textToFormattedHTML(event.data.data.response)
        responseDiv.parentElement.svgOn()
        // console.dir(event.data.data)
        if (isDone) {
            vscode.getState().history.push({ role: 'assistant', content: event.data.data.response })
            responseDiv.setAttribute('data-id', vscode.getState().history.length)
          renderMessagesFromState(vscode.getState())
        }
        microlightReset()
      }
      
    });
  
  //for autoscroll
  let height = window.document.documentElement.scrollHeight
  let fixedToBottom = window.innerHeight + window.document.documentElement.scrollTop >= window.document.documentElement.scrollHeight - 3
  const observer = new MutationObserver(function (mutations) {
    if(window.document.documentElement.scrollHeight > height && fixedToBottom) {
      window.scrollTo(0, document.body.scrollHeight)
    }
    height = window.document.documentElement.scrollHeight
    // console.log('Element height changed:', window.document.documentElement.scrollHeight);
  });
  const observerConfig = { attributes: true, childList: true, subtree: true };
  observer.observe(window.document, observerConfig)

  window.addEventListener('scroll', function() {
    fixedToBottom = Boolean(this.innerHeight + this.document.scrollingElement.scrollTop >= this.document.scrollingElement.scrollHeight - 3)
  });

  })();
  

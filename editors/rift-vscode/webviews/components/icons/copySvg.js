export function copySvg(color = "var(--vscode-panelTitle-inactiveForeground)") {
  const svgNS = "http://www.w3.org/2000/svg";

  // Create the <svg> element
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("fill", color);
  svg.setAttribute("height", "20");
  svg.setAttribute("width", "20");
  svg.setAttribute("version", "1.1");
  svg.setAttribute("id", "Capa_1");
  svg.setAttribute("viewBox", "0 0 352.804 352.804");
  svg.setAttribute("xml:space", "preserve");

  // Create the <g> element
  const g = document.createElementNS(svgNS, "g");
  svg.appendChild(g);

  // Create the <path> element
  const path = document.createElementNS(svgNS, "path");
  path.setAttribute(
    "d",
    "M318.54,57.282h-47.652V15c0-8.284-6.716-15-15-15H34.264c-8.284,0-15,6.716-15,15v265.522c0,8.284,6.716,15,15,15h47.651 v42.281c0,8.284,6.716,15,15,15H318.54c8.284,0,15-6.716,15-15V72.282C333.54,63.998,326.824,57.282,318.54,57.282z M49.264,265.522V30h191.623v27.282H96.916c-8.284,0-15,6.716-15,15v193.24H49.264z M303.54,322.804H111.916V87.282H303.54V322.804 z",
  );
  g.appendChild(path);

  return svg;
}

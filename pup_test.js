const Puppeteer = require('puppeteer');

(async () => {
  const browser = await Puppeteer.launch();
  const page = await browser.newPage();
  await page.goto('https://www.nickmvincent.com');

  const getPos = (links) => {
    let ret = []
    links.forEach((link) => {
        const {top, left, bottom, right} = link.getBoundingClientRect();
        ret.push({'href': link.href, top, left, bottom, right});
    });
    return ret;
  };
  const linkRects = await page.$$eval('a', getPos);
//   const rect = await page.evaluate(links.forEach((link) => {
//     console.log(link)
//     const {top, left, bottom, right} = link.getBoundingClientRect();
//     return {top, left, bottom, right};
//   }), links);
  console.log(linkRects);

  await browser.close();
})();
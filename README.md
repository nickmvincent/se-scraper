# About this repo
This a fork of https://github.com/NikolaiT/se-scraper.

This fork has various modifications designed for a research project studying the incidence of Wikipedia links in SERPs.

This fork is not kept up to date with the original se-scraper, so you may want to check out the repo for technical notes, new features, advanced usage etc.

A major contribution of the paper is "spatial analysis" of SERPs that considers the coordinates of link elements within a page (i.e. the coordinates of the top left corner of a link elements).

For additional work on the topic of spatial analysis of SERP data, I am working on a new library that is heavily inspired by se-scraper and puppeteer-based data collection but has a separete codebase. I recommend starting with either `se-scraper` (above) or the new `LinkCoordMin` code (below) instead, as this repo exists primarily as a companion to the WikiWorkshop 2020 non-archival paper (http://www.nickmvincent.com/static/WikiSerp2020.pdf).

LinkCoordMin:

https://github.com/nickmvincent/LinkCoordMin

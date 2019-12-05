# Abstract


# Intro
Previous work has highlighted critical interdependencies between Wikipedia and Google's search engine. For many queries, Wikipedia links appear more than any other website on Google search engine results pages (SERPs) (cite) and removing Wikipedia links would lower Google's click-through-rate, any important search metric. Conversely, this means Google is a key source of traffic to Wikipedia.

This relationship likely extends to other search engines, although this has not yet been explicitly studied (notably, DuckDuckGo includes a feature that allows users to search directly on Wikipedia -- just add "!w" to a query). In this work, we perform an audit style experiment that further characterizes the role of Wikipedia article in serving search queries for a variety of search engines.

This work differs from previous work in x ways:
* We consider three popular search engines: Google, Bing, and DuckDuckGo.
* In contrast to previous work that We characterize importance using the location of Wikipedia links within the Cartesian space of a SERP. Then, by proposing several models of user click behavior, we produce a lower bound and upper bound for the importance of Wikipedia to serving the search query.
* We examine how Wikipedia's appearance differs across types of queries (e.g. blue link vs. knowledge panel vs. answer box)

Querysets:
* 2018 year in review
* Geoquery Q&A dataset
* Bing Health query dataset




... We assume that every link on the SERP "answers" the query for at least one user (any link which does not would be eventually filtered out through training).


const se_scraper = require('./index.js');

const mkdirp = require('mkdirp');

const devices = [
    'desktop',
    'mobile',
];
const search_engines = [
    'google',
    'bing',
    'duckduckgo',
];
const query_lists = [
    'trend_sample',

]
const configs = [];

for (const device of devices) {
    for (const search_engine of search_engines) {
        for (const queries of query_lists) {
            configs.push({
                device,
                search_engine,
                queries
            })
        }
    }
}

console.log(configs)
for (const config of configs) {
    let output_dir = `output/${config.device}/${config.search_engine}/${config.queries}`;
    mkdirp(output_dir)

    let user_agent = '';
    const keyword_file = `search_queries/prepped/${config.queries}.txt`;

    if (config.device === 'mobile') {
        user_agent = 'Mozilla/5.0 (iPhone; CPU iPhone OS 10_3_1 like Mac OS X) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.0 Mobile/14E304 Safari/602.1';
    } else {
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36';
    }

    const output_file = `${output_dir}/results.json`;

    var fs = require("fs");
    console.log(config.queries, keyword_file)
    var text = fs.readFileSync(keyword_file, "utf-8");
    var keywords = text.split("\n")
    keywords = keywords.filter(Boolean)

    let browser_config = {
        user_agent: user_agent,
        mobile: config.device === 'mobile',
        // if random_user_agent is set to True, a random user agent is chosen
        random_user_agent: false,
        // whether to start the browser in headless mode
        headless: true,
        // whether debug information should be printed
        // level 0: print nothing
        // level 1: print most important info
        // ...
        // level 4: print all shit nobody wants to know
        debug_level: 1,
        // specify flags passed to chrome here
        chrome_flags: [],
        // path to js module that extends functionality
        // this module should export the functions:
        // get_browser, handle_metadata, close_browser
        // must be an absolute path to the module
        //custom_func: resolve('examples/pluggable.js'),
        custom_func: '',
        // use a proxy for all connections
        // example: 'socks5://78.94.172.42:1080'
        // example: 'http://118.174.233.10:48400'
        proxy: '',
        // a file with one proxy per line. Example:
        // socks5://78.94.172.42:1080
        // http://118.174.233.10:48400
        proxy_file: '',
        puppeteer_cluster_config: {
            timeout: 10 * 60 * 1000, // max timeout set to 10 minutes
            monitor: false,
            concurrency: 1, // one scraper per tab
            maxConcurrency: 1, // scrape with 1 tab
        },
        screen_output: true,
        html_output: true,
    };

    (async () => {
        // scrape config can change on each scrape() call
        let scrape_config = {
            // which search engine to scrape
            search_engine: config.search_engine,
            // the number of pages to scrape for each keyword
            num_pages: 1,
            // OPTIONAL PARAMS BELOW:
            // google_settings: {
            //     gl: 'us', // The gl parameter determines the Google country to use for the query.
            //     hl: 'fr', // The hl parameter determines the Google UI language to return results.
            //     start: 0, // Determines the results offset to use, defaults to 0.
            //     num: 100, // Determines the number of results to show, defaults to 10. Maximum is 100.
            // },
            // how long to sleep between requests. a random sleep interval within the range [a,b]
            // is drawn before every request. empty string for no sleeping.
            keywords: keywords,
            sleep_range: '',
            // path to output file, data will be stored in JSON
            output_file: output_file,
            // whether to prevent images, css, fonts from being loaded
            // will speed up scraping a great deal
            block_assets: false,
            // check if headless chrome escapes common detection techniques
            // this is a quick test and should be used for debugging
            test_evasion: false,
            apply_evasion_techniques: true,
            // log ip address data
            log_ip_address: false,
            // log http headers
            log_http_headers: false,
        };

        let results = await se_scraper.scrape(browser_config, scrape_config);
        //console.dir(results, {depth: null, colors: true});
    })();
}
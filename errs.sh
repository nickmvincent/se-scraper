/usr/bin/time -v node driver.js desktop duckduckgo errs_desktop_duckduckgo_top &> logs/errs_desktop_duckduckgo_top.txt
/usr/bin/time -v node driver.js mobile google errs_mobile_google_top &> logs/errs_mobile_google_top.txt
/usr/bin/time -v node driver.js mobile bing errs_mobile_bing_top &> logs/errs_mobile_bing_top.txt
/usr/bin/time -v node driver.js mobile duckduckgo errs_mobile_duckduckgo_top &> logs/errs_mobile_duckduckgo_top.txt
#/usr/bin/time -v node driver.js desktop bing errs4_desktop_bing_trend &> logs/errs4_desktop_bing_trend.txt
/usr/bin/time -v node driver.js desktop duckduckgo errs4_desktop_duckduckgo_trend &> logs/errs4_desktop_duckduckgo_trend.txt
#/usr/bin/time -v node driver.js mobile google errs4_mobile_google_top &> logs/errs4_mobile_google_top.txt
#/usr/bin/time -v node driver.js mobile bing errs4_mobile_bing_top &> logs/errs4_mobile_bing_top.txt
/usr/bin/time -v node driver.js mobile duckduckgo errs4_mobile_duckduckgo_trend &> logs/errs4_mobile_duckduckgo_trend.txt
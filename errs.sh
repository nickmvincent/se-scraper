/usr/bin/time -v node driver.js desktop google errs3_desktop_google_trend &> logs/errs3_desktop_google_trend.txt
/usr/bin/time -v node driver.js desktop bing errs3_desktop_bing_trend &> logs/errs3_desktop_bing_trend.txt
/usr/bin/time -v node driver.js desktop duckduckgo errs3_desktop_duckduckgo_trend &> logs/errs3_desktop_duckduckgo_trend.txt
/usr/bin/time -v node driver.js mobile google errs3_mobile_google_top &> logs/errs3_mobile_google_top.txt
/usr/bin/time -v node driver.js mobile google errs3_mobile_google_trend &> logs/errs3_mobile_google_trend.txt
/usr/bin/time -v node driver.js mobile bing errs3_mobile_bing_top &> logs/errs3_mobile_bing_top.txt
/usr/bin/time -v node driver.js mobile bing errs3_mobile_bing_trend &> logs/errs3_mobile_bing_trend.txt
/usr/bin/time -v node driver.js mobile duckduckgo errs3_mobile_duckduckgo_trend &> logs/errs3_mobile_duckduckgo_trend.txt
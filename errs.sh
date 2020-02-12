/usr/bin/time -v node driver.js desktop duckduckgo top  &> logs/errs_desktop_duckduckgo_top.txt
/usr/bin/time -v node driver.js mobile google top  &> logs/errs_mobile_google_top.txt
/usr/bin/time -v node driver.js mobile bing top  &> logs/errs_mobile_bing_top.txt
/usr/bin/time -v node driver.js mobile duckduckgo top  &> logs/errs_mobile_duckduckgo_top.txt
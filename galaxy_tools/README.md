# SPAC Galaxy Tools
   
   ## Requirements
   - Galaxy instance with Docker enabled
   - Docker image: nciccbr/spac:v1
   
   ## Installation
   1. Pull Docker image: `docker pull nciccbr/spac:v1`
   2. Copy tool directory to Galaxy's tools folder
   3. Add to tool_conf.xml:
```xml
      <tool file="spac_boxplot/spac_boxplot.xml"/>
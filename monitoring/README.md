# Monitoring

Asset-health observability for the Predictive Maintenance Platform, built on
**Grafana + InfluxDB**.

## Dashboards

[`grafana_dashboard.json`](./grafana_dashboard.json) provisions six panels:

| Panel | Purpose |
| --- | --- |
| Fleet RUL (P50) by Asset | Median Remaining Useful Life trend per asset |
| RUL Confidence Band | P10 / P50 / P90 interval from the TFT |
| Anomaly Score Heatmap | Per-asset Anomaly Transformer scores |
| Assets Below 48h Horizon | Count of CMMS work orders triggered |
| Physics vs ML Agreement | Digital-twin consistency gauge |
| OEE Impact | Estimated Overall Equipment Effectiveness uplift |

## Quick start

```bash
docker compose -f deployment/docker-compose.yml up influxdb grafana
# Grafana → http://localhost:3000  (admin / admin)
# InfluxDB → http://localhost:8086
```

The dashboard auto-loads from the provisioning mount defined in
[`deployment/docker-compose.yml`](../deployment/docker-compose.yml).

## Alerting

RUL-below-threshold and anomaly-spike alerts route to **PagerDuty + Slack**.
Configure contact points in Grafana Alerting and set the rule condition to
`rul_p50 < 48` or `anomaly_score > 0.55`.

import type {
  AlpacaConnectivityStatus,
  DataBackboneStatus,
  FactorStrategyReport,
  MicrostructureReport,
  MLStrategyReport,
  MomentumStrategyReport,
  PairsStrategyReport,
  PeadStrategyReport,
  RiskReport,
  VolRelValReport,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  dataStatus: () => getJson<DataBackboneStatus>("/api/data/status"),
  pairsReport: () => getJson<PairsStrategyReport>("/api/pairs/report"),
  factorsReport: () => getJson<FactorStrategyReport>("/api/factors/report"),
  mlReport: () => getJson<MLStrategyReport>("/api/ml/report"),
  momentumReport: () => getJson<MomentumStrategyReport>("/api/momentum/report"),
  peadReport: () => getJson<PeadStrategyReport>("/api/pead/report"),
  riskReport: () => getJson<RiskReport>("/api/risk/report"),
  microstructureReport: () => getJson<MicrostructureReport>("/api/microstructure/report"),
  volRelValReport: () => getJson<VolRelValReport>("/api/vol-relval/report"),
  alpacaStatus: () => getJson<AlpacaConnectivityStatus>("/api/execution/status"),
};

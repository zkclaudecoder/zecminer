# Zcash (Equihash 200,9) GPU Miner Research — April 2026

## CRITICAL FINDING: GPU Mining Viability

**Zcash Equihash 200,9 is dominated by ASICs in 2026.** ASIC miners like the Bitmain
Antminer Z15 Pro (840 KSol/s, ~2560W) earn roughly $56/day. The entire Zcash network
hashrate is ~15.68 GS/s, almost entirely ASIC-driven. A high-end GPU producing ~800-1500
Sol/s would represent a negligible fraction of network hashrate.

**GPU mining ZEC is not profitable in 2026 at normal electricity rates.** This setup
may still be useful for learning, testing, hobby mining, or mining other Equihash-variant
coins (Bitcoin Gold, Ycash, etc.) that remain GPU-friendly.

---

## Zcash Network Status (April 2026)

- **Consensus**: Still Proof-of-Work (Equihash 200,9)
- **NU6**: Activated November 2024 (funding model changes, not PoS)
- **NU6.1**: Activated November 2025
- **Proof-of-Stake**: NOT yet implemented. The "Crosslink" hybrid PoW/PoS proposal
  (Shielded Labs) has testnet deployment expected in 2026. A coinholder sentiment poll
  was scheduled for January 2026 to inform NU7 priorities.
- **Block reward**: 1.5625 ZEC per block (post-2024 halving)
- **Next halving**: Expected late 2028
- **Network hashrate**: ~15.68 GS/s
- **Network difficulty**: ~138,783,153

---

## Miner Evaluations

### 1. miniZ

| Field | Details |
|-------|---------|
| Latest version | v2.5e3 |
| Status | **Actively maintained** |
| Open/Closed source | Closed source |
| Equihash 200,9 support | **NO** — supports 125,4 / 144,5 / 150,5 / 150,5,3 / 192,7 / 210,9 / 96,5 only |
| RTX 5090/Blackwell | Yes, RTX 50XX support added in recent releases |
| Developer fee | 2% (Equihash algorithms) |
| Website | https://miniz.cc/ |
| GitHub | https://github.com/miniZ-miner/miniZ/releases |

**Verdict**: Cannot mine Zcash (ZEC). Only supports modified Equihash variants used by
other coins (Bitcoin Gold, Beam, Flux/ZelCash, etc.). Excellent for those coins on RTX 5090.

---

### 2. GMiner

| Field | Details |
|-------|---------|
| Latest version | v3.44 |
| Status | **Actively maintained** |
| Open/Closed source | Closed source |
| Equihash 200,9 support | **Unclear/Limited** — supports Equihash 144,5 / 192,7 / 210,9 / 96,5. Earlier versions supported 200,9 but current focus is on other algorithms |
| RTX 5090/Blackwell | Not explicitly confirmed but likely (supports modern CUDA) |
| Developer fee | 2% (Equihash), 0.65% (Ethash), 1% (KAWPOW) |
| GitHub | https://github.com/develsoftware/GMinerRelease |
| Website | https://gminer.info/ |

**Verdict**: May have residual Equihash 200,9 support but it is not a focus. Better suited
for other algorithms on modern hardware.

---

### 3. lolMiner

| Field | Details |
|-------|---------|
| Latest version | v1.98a |
| Status | **Actively maintained** |
| Open/Closed source | Closed source |
| Equihash 200,9 support | **NO** — supports Equihash 144,5 / 192,7 and Beam/Cuckoo variants |
| RTX 5090/Blackwell | Yes, RTX 5000 series support added |
| Developer fee | ~1-2% depending on algorithm |
| GitHub | https://github.com/Lolliedieb/lolMiner-releases |
| Website | https://lolminer.site/ |

**Verdict**: Cannot mine Zcash (ZEC). Focused on Ethash, Etchash, and non-200,9 Equihash
variants. Good miner for other coins.

---

### 4. EWBF's CUDA Equihash Miner

| Field | Details |
|-------|---------|
| Latest version | v0.6 (also v0.3.4b for original Zcash miner) |
| Status | **ABANDONED** — no updates since ~2019 |
| Open/Closed source | Closed source |
| Equihash 200,9 support | v0.3.4b supported ZEC (200,9); v0.6 only supports 144,5 / 192,7 / 210,9 / 96,5 |
| RTX 5090/Blackwell | **No** — will not work on modern GPUs |
| Developer fee | 2% |
| GitHub | https://github.com/zhashpro/EWBF-0.3.4b |

**Verdict**: Dead project. Will not compile or run on RTX 5090. Do not use.

---

### 5. Funakoshi Miner

| Field | Details |
|-------|---------|
| Latest version | v5.2 |
| Status | **Unclear** — some updates but not actively maintained for modern hardware |
| Open/Closed source | Closed source (GitHub releases only) |
| Equihash 200,9 support | **YES** — supports both Equihash 200,9 and 144,5 |
| RTX 5090/Blackwell | **Unknown/Unlikely** — last significant CUDA updates predate Blackwell |
| Developer fee | 0% (claims no fee) |
| Historical performance | ~525 Sol/s on GTX 1080 (200,9) |
| GitHub | https://github.com/funakoshi2718/funakoshi-miner |

**Verdict**: One of the few miners that explicitly supports Equihash 200,9. However,
unlikely to work on RTX 5090 without CUDA kernel updates for SM 12.0. Worth checking
if recent builds exist.

---

### 6. GMO Cryptknocker

| Field | Details |
|-------|---------|
| Status | Claimed to be "fastest" Equihash 200,9 Nvidia GPU miner |
| Open/Closed source | Closed source |
| Equihash 200,9 support | **YES** |
| RTX 5090/Blackwell | **Unknown** — unclear if still maintained |
| Developer fee | 2% |
| Platforms | Windows and Linux |

**Verdict**: Supports Equihash 200,9 but maintenance status is uncertain. Research further
before using.

---

### 7. nheqminer (current project)

| Field | Details |
|-------|---------|
| Latest version | From 2018 (NiceHash) |
| Status | **Abandoned upstream** — you are modernizing it locally |
| Open/Closed source | Open source (MIT) |
| Equihash 200,9 support | **YES** |
| RTX 5090/Blackwell | Requires CUDA kernel modernization (in progress) |
| Developer fee | 0% |
| GitHub | https://github.com/nicehash/nheqminer |

**Verdict**: The approach of modernizing nheqminer CUDA kernels for SM 12.0 is actually
one of the most viable paths, since most commercial miners have dropped Equihash 200,9
support entirely.

---

### 8. Optiminer Equihash

| Field | Details |
|-------|---------|
| Status | Last updated ~2018 |
| Open/Closed source | Closed source |
| Equihash 200,9 support | **YES** — unified [200,9] / [192,7] / [96,6] |
| RTX 5090/Blackwell | **No** — Linux only, primarily AMD GPUs |
| Developer fee | 1% |
| GitHub | https://github.com/pcollinsca47/OptiminerEquihash |

**Verdict**: Dead project. Linux/AMD only. Not applicable.

---

### 9. SilentArmy

| Field | Details |
|-------|---------|
| Status | Last updated ~2017 |
| Open/Closed source | Open source |
| Equihash 200,9 support | **YES** |
| RTX 5090/Blackwell | **No** — OpenCL based, very old |
| GitHub | https://github.com/mbevand/silentarmy |

**Verdict**: Dead project. OpenCL only. Not applicable.

---

## RTX 5090 Expected Hashrate on Equihash 200,9

No published benchmarks exist for RTX 5090 on Equihash 200,9 because:
1. Most GPU miners have dropped Equihash 200,9 support
2. ASIC dominance means few people GPU-mine ZEC

**Estimates based on scaling from older GPUs:**
- GTX 1070: ~400 Sol/s (nheqminer benchmark)
- GTX 1080: ~525 Sol/s (Funakoshi benchmark)
- RTX 4090: ~1500 Sol/s (general Equihash, unconfirmed for 200,9 specifically)
- **RTX 5090: ~1800-2500 Sol/s (estimated)** — based on ~1.5-1.7x scaling from RTX 4090
  due to higher CUDA core count (21,760 vs 16,384) and GDDR7 bandwidth

For comparison, the Antminer Z15 Pro ASIC does 840,000 Sol/s (840 KSol/s). A single RTX
5090 would be roughly 0.15-0.3% of one ASIC's hashrate.

---

## Mining Pools (Zcash / ZEC)

### 1. Flypool (Bitfly/Ethermine)

| Field | Details |
|-------|---------|
| URL | https://zcash.flypool.org/ |
| Fee | 1% (PPLNS) |
| Min payout | Customizable (default ~0.01 ZEC) |
| Payout frequency | Varies by threshold |
| Servers | US, Europe, Asia |
| Stratum | `stratum+tcp://zec.flypool.org:3333` (default) |
| Stratum (SSL) | `stratum+ssl://zec.flypool.org:3443` |
| US server | `stratum+tcp://us1-zec.flypool.org:3333` |
| EU server | `stratum+tcp://eu1-zec.flypool.org:3333` |
| Asia server | `stratum+tcp://asia1-zec.flypool.org:3333` |

### 2. 2Miners

| Field | Details |
|-------|---------|
| URL | https://2miners.com/zec-mining-pool |
| Fee | 0.5% (PPLNS), 1.5% (Solo) |
| Min payout | 0.1 ZEC |
| Payout frequency | Every 2 hours |
| Servers | EU, US, Asia (DDOS protected) |
| Stratum | `stratum+tcp://zec.2miners.com:1010` |
| US server | `stratum+tcp://us-zec.2miners.com:1010` |
| Alt port | `:1111` |
| NiceHash compatible | Yes |

### 3. NanoPool

| Field | Details |
|-------|---------|
| URL | https://zec.nanopool.org/ |
| Fee | 1% |
| Min payout | 0.01 ZEC |
| Stratum | `stratum+tcp://zec-eu1.nanopool.org:6666` |
| US server | `stratum+tcp://zec-us-east1.nanopool.org:6666` |
| Asia server | `stratum+tcp://zec-asia1.nanopool.org:6666` |

### 4. AntPool

| Field | Details |
|-------|---------|
| URL | https://www.antpool.com/ |
| Fee | 0% (claimed) |
| Note | Large pool, primarily ASIC miners |

### 5. MiningPoolHub

| Field | Details |
|-------|---------|
| Fee | 0.9% |
| Note | Multi-coin pool with auto-exchange |

---

## Recommendations

### If the goal is to mine ZEC specifically on RTX 5090:

1. **Continue modernizing nheqminer** — It is open source, supports Equihash 200,9, and
   you have already done significant work updating CUDA kernels for SM 12.0. This is
   actually one of the best paths since commercial miners have abandoned 200,9.

2. **Check Funakoshi Miner** — It explicitly supports Equihash 200,9 with CUDA. Test
   whether it launches on RTX 5090 (it may need CUDA compatibility updates).

3. **Accept low profitability** — Even with a working miner, RTX 5090 GPU mining of ZEC
   is a hobby/learning exercise, not a profit center in 2026.

### If the goal is profitable GPU mining:

Consider mining Equihash-variant coins (Flux, Bitcoin Gold, Ycash) or other algorithms
(KawPow, Ethash, Autolykos) with miniZ, lolMiner, or GMiner, which all have excellent
RTX 5090 support.

---

## Sources

- [miniZ official site](https://miniz.cc/)
- [miniZ GitHub releases](https://github.com/miniZ-miner/miniZ/releases)
- [GMiner GitHub](https://github.com/develsoftware/GMinerRelease)
- [GMiner official site](https://gminer.info/)
- [lolMiner GitHub](https://github.com/Lolliedieb/lolMiner-releases)
- [lolMiner official site](https://lolminer.site/)
- [EWBF GitHub](https://github.com/zhashpro/EWBF-0.3.4b)
- [Funakoshi GitHub](https://github.com/funakoshi2718/funakoshi-miner)
- [nheqminer GitHub](https://github.com/nicehash/nheqminer)
- [Optiminer GitHub](https://github.com/pcollinsca47/OptiminerEquihash)
- [SilentArmy GitHub](https://github.com/mbevand/silentarmy)
- [Flypool Zcash](https://zcash.flypool.org/)
- [2Miners ZEC pool](https://2miners.com/zec-mining-pool)
- [Zcash ZIP 253 (NU6)](https://zips.z.cash/zip-0253)
- [Zcash Protocol Spec NU6.1](https://zips.z.cash/protocol/protocol.pdf)
- [minerstat RTX 5090](https://minerstat.com/hardware/nvidia-geforce-rtx-5090)
- [Kryptex RTX 5090](https://www.kryptex.com/en/hardware/nvidia-rtx-5090)
- [CoinCodex Zcash Mining 2026](https://coincodex.com/article/75616/how-to-mine-zcash/)
- [CoinBureau Zcash Pools](https://coinbureau.com/mining/best-zcash-pools/)
- [MiningPoolStats Zcash](https://miningpoolstats.net/coins/zcash/)

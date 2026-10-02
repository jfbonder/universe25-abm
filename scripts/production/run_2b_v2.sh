#!/usr/bin/env bash
# Orquestador de los barridos de producción (results_v2/f2b_*): diseños, corridas y análisis por etapa.
#
#   RES=results_v2 FG=1 bash scripts/production/run_2b_v2.sh     # en primer plano (uso previsto)
#   bash scripts/production/run_2b_v2.sh                         # se relanza con nohup y vuelve enseguida
#   STAGES="base nulos rob" FG=1 bash ...                       # subconjunto, en ese orden
#   SMOKE=1 FG=1 bash ...                                       # humo: todas las etapas recortadas (2 réplicas, T <= 250)
#   SMOKE=1 STAGES=p1 SMOKE_POINTS=0,6,56,62 FG=1 bash ...      # humo de una etapa
#
# Variables: PRESET (calhoun_C1), RES (results_v2; relativo a la raíz si no empieza con /), SUFFIX ("" para C1),
#   STAGES (orden del manifiesto), N_JOBS (4), DEADLINE_H (5), REGEN (1 = regenerar designs_v2/<PRESET>/),
#   SMOKE/SMOKE_REPS/SMOKE_T/SMOKE_POINTS, MODEL (v2: f2b_model_v2.U2v2).
# Reanudable: relanzar el mismo comando retoma cada etapa donde quedó (u25 sweep saltea los pares hechos; los
#   transplantes, los brazos con O12.txt). Un segundo orquestador sobre el mismo RES/SUFFIX aborta (flock).
# Corte: al vencer el plazo un vigía manda SIGTERM al runner (lo escrito queda); P2/P3 se saltean si no entran;
#   Sobol baja réplicas hasta reps_min.
set -u
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
MAT=$ROOT/scripts/production          # orquestador, runner, diseños y evaluador
ANA=$ROOT/scripts/analysis            # pipeline estadístico (python -m pipeline)
PRESET=${PRESET:-calhoun_C1}
TAG=${PRESET#calhoun_}
SMOKE=${SMOKE:-0}
MODEL=${MODEL:-v2}
if [ -z "${F2B_CHILD:-}" ]; then
  if [ -z "${SUFFIX+x}" ]; then
    if [ "$PRESET" = "calhoun_C1" ]; then SUFFIX=""; else SUFFIX="_$TAG"; fi
  fi
  [ "$SMOKE" = "1" ] && SUFFIX="${SUFFIX}_smoke"
fi
RES=${RES:-results_v2}
case "$RES" in /*) ;; *) RES=$ROOT/$RES ;; esac
STATE=$RES/f2b_estado$SUFFIX
N_JOBS=${N_JOBS:-4}
DEADLINE_H=${DEADLINE_H:-5}
DEADLINE=${DEADLINE:-$(( $(date +%s) + DEADLINE_H * 3600 ))}
DES=${DES:-$MAT/designs_v2/$PRESET}
export PRESET SUFFIX SMOKE RES STATE N_JOBS DEADLINE DES MODEL
export PYTHONPATH=$ROOT:$MAT:$ANA${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-python3}
GEN="$PY $MAT/gen_designs_v2.py"
V2_STAGES=" nulos rob robmu ci size q7aw lhsx trans "      # etapas con análisis de pipeline/v2.py
CHECKPOINTS=${CHECKPOINTS:-" abl2 q7aw sobol "}             # informes consolidados además del final

mkdir -p "$STATE"
if [ -z "${F2B_CHILD:-}" ] && [ "${FG:-0}" != "1" ]; then
  LOG=$STATE/run_2b_v2_$(date +%Y%m%d_%H%M%S).log
  F2B_CHILD=1 nohup bash "$0" "$@" > "$LOG" 2>&1 < /dev/null &
  echo "run_2b_v2.sh en segundo plano: PID $!  log: $LOG  plazo: $(date -d @"$DEADLINE" '+%F %T')"
  exit 0
fi

exec 9> "$STATE/.lock"
if ! flock -n 9; then echo "otro run_2b_v2.sh usa $STATE (lock); salgo"; exit 1; fi
log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATE/orquestador.log"; }
log "inicio V2: PRESET=$PRESET RES=$RES SUFFIX='$SUFFIX' N_JOBS=$N_JOBS plazo=$(date -d @"$DEADLINE" '+%F %T') SMOKE=$SMOKE"
(cd "$ROOT" && git rev-parse --short HEAD 2>/dev/null | sed 's/^/commit /'; git status --short u25 2>/dev/null | sed 's/^/u25 sin commitear: /') >> "$STATE/orquestador.log"

# ---------------------------------------------------------------- diseños y verificación previa
if [ ! -f "$DES/manifest.json" ] || [ "${REGEN:-0}" = "1" ]; then
  log "generando diseños V2 para $PRESET en $DES"
  $GEN all --preset "$PRESET" --out "$DES" >> "$STATE/orquestador.log" 2>&1 || { log "ERROR al generar diseños"; exit 2; }
fi
if [ "$SMOKE" = "1" ]; then             # el humo no toca los diseños adaptativos reales (refine, lhsx, sobol_auto)
  rm -rf "$STATE/des_smoke"; mkdir -p "$STATE/des_smoke"
  for d in "$DES"/*; do case "$(basename "$d")" in refine|lhsx|sobol_auto) ;; *) cp -r "$d" "$STATE/des_smoke/" ;; esac; done
  DES=$STATE/des_smoke
fi
if ! $GEN check --preset "$PRESET" --des "$DES" >> "$STATE/orquestador.log" 2>&1; then
  log "ERROR: la verificación de diseños falló (ver orquestador.log)"; exit 2
fi
PEND=$($PY -c "import json;m=json.load(open('$DES/manifest.json'));print(len(m.get('pendiente_api',[])))")
[ "$PEND" != "0" ] && log "AVISO: $PEND brazos pendientes de API (manifest.json: pendiente_api); corren sin ellos"
# verificación de la API del evaluador: el primario tiene que existir en esta versión de u25 (si no, cada corrida
# daría ERR); una variante que falta sólo deja sus columnas como ERR
if ! (cd "$MAT" && $PY "$MAT/api_check_v2.py" >> "$STATE/orquestador.log" 2>&1); then
  log "ERROR: la API del evaluador no soporta el primario de v2_api.json (ver orquestador.log)"; exit 3
fi
if [ -z "${STAGES:-}" ]; then
  STAGES=$($PY -c "import json;print(' '.join(json.load(open('$DES/manifest.json'))['order']))")
fi
log "etapas: $STAGES"

# ---------------------------------------------------------------- vigía del plazo
run_watched() {   # run_watched LOGFILE cmd... ; devuelve el código del comando (124 si se cortó por plazo)
  local lf=$1; shift
  "$@" >> "$lf" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2> /dev/null; do
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
      log "plazo vencido: SIGTERM a $pid (la etapa queda reanudable)"
      kill -TERM "$pid" 2> /dev/null; wait "$pid"; return 124
    fi
    sleep 10
  done
  wait "$pid"
}

scale() { cat "$STATE/scale" 2> /dev/null || echo 1.0; }

analyze() {   # analyze STAGE ANALYSIS OUT
  local st=$1 an=$2 out=$3
  log "análisis de $st -> $out/analisis"
  if [ "$st" = "trans" ]; then
    (cd "$ANA" && $PY -m pipeline v2 trans "$out") >> "$STATE/analisis.log" 2>&1 || log "análisis v2 de $st con errores"
    (cd "$ANA" && $PY -m pipeline f2b trans "$out" --out "$out/analisis_v1") >> "$STATE/analisis.log" 2>&1 || true
  elif [[ "$V2_STAGES" == *" $st "* ]]; then
    (cd "$ANA" && $PY -m pipeline v2 "$st" "$out") >> "$STATE/analisis.log" 2>&1 || log "análisis v2 de $st con errores"
  else
    (cd "$ANA" && $PY -m pipeline f2b "$an" "$out") >> "$STATE/analisis.log" 2>&1 || log "análisis de $st con errores"
  fi
  # los informes consolidados (lentos: bootstraps) corren sólo en los puntos de control y al final
  [[ " $CHECKPOINTS " == *" $st "* ]] && final_reports
  return 0
}

final_reports() {
  (cd "$ANA" && $PY -m pipeline f2b final "$RES" --suffix "$SUFFIX") >> "$STATE/analisis.log" 2>&1 || log "informe Q con errores"
  # la familia Q1–Q7 y QC también con el evaluador v1 completo (prerregistro; tabla de desvíos)
  (cd "$ANA" && F2B_VARIANT=v1 $PY -m pipeline f2b final "$RES" --suffix "$SUFFIX" --out "$RES/f2b_final$SUFFIX/v1") \
      >> "$STATE/analisis.log" 2>&1 || log "informe Q (evaluador v1) con errores"
  (cd "$ANA" && $PY -m pipeline v2 final "$RES" --suffix "$SUFFIX") >> "$STATE/analisis.log" 2>&1 || log "informe V2 con errores"
}

# ---------------------------------------------------------------- una etapa
run_stage() {
  local st=$1 sdir out reps_override=""
  sdir=$DES/$st
  out=$RES/f2b_$st$SUFFIX
  case "$st" in
    sobol)
      if [ -f "$out/.design_src" ]; then sdir=$(cat "$out/.design_src"); case "$sdir" in /*) ;; *) sdir=$ROOT/$sdir ;; esac
      else
        local auto
        auto=$($GEN sobol-from-morris --preset "$PRESET" --des "$DES" --morris "$RES/f2b_morris$SUFFIX" 2>> "$STATE/orquestador.log" | tail -1)
        [ -n "$auto" ] && [ -d "$auto" ] && sdir=$auto
      fi ;;
    sobolx)
      out=$RES/f2b_sobol$SUFFIX
      [ -f "$out/.design_src" ] || { log "sobolx: Sobol no empezó; salteo"; return 0; }
      sdir=$(cat "$out/.design_src"); case "$sdir" in /*) ;; *) sdir=$ROOT/$sdir ;; esac
      reps_override=$($PY -c "import json;print(json.load(open('$sdir/stage.json')).get('ext_reps',6))") ;;
    refine)
      local runs=""
      for p in p1 p2 p1aw p5; do [ -d "$RES/f2b_$p$SUFFIX/analisis" ] && runs="$runs $RES/f2b_$p$SUFFIX"; done
      [ -f "$DES/refine/stage.json" ] || $GEN refine --preset "$PRESET" --des "$DES" --runs $runs >> "$STATE/orquestador.log" 2>&1
      [ -f "$DES/refine/stage.json" ] || { log "refine: sin puntos de frontera; salteo"; return 0; } ;;
    lhsx)
      local runs=""
      for p in lhs lhs2; do [ -d "$RES/f2b_$p$SUFFIX/analisis" ] && runs="$runs $RES/f2b_$p$SUFFIX"; done
      [ -f "$DES/lhsx/stage.json" ] || $GEN lhsx --preset "$PRESET" --des "$DES" --runs $runs >> "$STATE/orquestador.log" 2>&1
      [ -f "$DES/lhsx/stage.json" ] || { log "lhsx: sin puntos frontera; salteo"; return 0; } ;;
  esac
  [ -f "$sdir/stage.json" ] || { log "etapa $st: no hay diseño en $sdir; salteo"; return 0; }
  if [ "$SMOKE" = "1" ]; then
    $GEN smoke --des "$(dirname "$sdir")" --stage "$(basename "$sdir")" --points "${SMOKE_POINTS:-}" \
        --reps "${SMOKE_REPS:-2}" --T "${SMOKE_T:-250}" --out "$STATE/smoke_des/$st" >> "$STATE/orquestador.log" 2>&1
    [ -f "$sdir/problem.json" ] && cp "$sdir/problem.json" "$STATE/smoke_des/$st/"
    sdir=$STATE/smoke_des/$st
  fi
  eval "$($GEN stagevars --des "$(dirname "$sdir")" --stage "$(basename "$sdir")")"
  local now remaining
  now=$(date +%s); remaining=$(( DEADLINE - now ))
  if [ "$remaining" -le 60 ]; then log "sin tiempo para $st"; return 99; fi
  eval "$($GEN budget --des "$(dirname "$sdir")" --stage "$(basename "$sdir")" --out "$out" \
          --n-jobs "$N_JOBS" --remaining "$remaining" --scale "$(scale)" ${reps_override:+--reps-override $reps_override})"
  log "etapa $st (P$PRIORITY, $KIND): $NPTS puntos x $REPS rép, T=$T, faltan $TODO, estimado $(( EST / 60 )) min (escala $SCALE); decisión $DECISION"
  if [ "$DECISION" = "SKIP" ]; then return 0; fi
  mkdir -p "$out"
  echo "${sdir#$ROOT/}" > "$out/.design_src"   # relativa a la raíz
  cp "$sdir/labels.json" "$out/labels.json"
  [ -f "$sdir/problem.json" ] && cp "$sdir/problem.json" "$out/problem.json"
  $PY - "$sdir/stage.json" "$out/stage.json" "$REPS" <<'EOF'
import json, sys
st = json.load(open(sys.argv[1])); st["reps_run"] = int(sys.argv[3]); json.dump(st, open(sys.argv[2], "w"), indent=1)
EOF
  local rc=0
  if [ "$DECISION" = "RUN" ]; then
    local preset_arg=() crn_arg=() full_arg=()
    [ -n "$SPRESET" ] && preset_arg=(--preset "$SPRESET") || preset_arg=(--preset "$PRESET")
    [ "$CRN" = "1" ] && crn_arg=(--crn)
    [ "$FULL" = "1" ] && full_arg=(--full-series)
    if [ "$KIND" = "transplant2" ]; then
      run_watched "$out/runner.log" $PY "$MAT/u25x.py" transplant2 --design "$sdir" --reps "$REPS" --out "$out" \
          "${preset_arg[@]}" --T-main "$TMAIN" --T-after "$TAFTER" --t-fix "$TFIX" --band "$BAND" \
          --cont-after "$CONT" --base-seed "$BASE_SEED" --n-jobs "$N_JOBS" --model "$MODEL"
      rc=$?
    elif [ "$KIND" = "transplant" ]; then
      run_watched "$out/runner.log" $PY "$MAT/u25x.py" transplant --design "$sdir" --reps "$REPS" --out "$out" \
          "${preset_arg[@]}" --T-main "$TMAIN" --T-after "$TAFTER" --base-seed "$BASE_SEED" --n-jobs "$N_JOBS"
      rc=$?
    else
      run_watched "$out/runner.log" $PY "$MAT/u25x.py" sweep --design "$sdir" --reps "$REPS" --T "$T" --out "$out" \
          "${preset_arg[@]}" "${crn_arg[@]}" "${full_arg[@]}" --series-every "$SERIES_EVERY" --base-seed "$BASE_SEED" \
          --n-jobs "$N_JOBS" --quiet --log-every 50 --model "$MODEL"
      rc=$?
      [ "$SERIES_EVERY" != "0" ] && $PY -m u25 collect "$out" >> "$out/runner.log" 2>&1
      $GEN measure --out "$out" --c-est "$CCPU" --state "$STATE/escalas.txt" > "$STATE/scale" 2>> "$STATE/orquestador.log"
    fi
    log "etapa $st: runner terminó con código $rc; escala de costo $(scale)"
  fi
  [ "$rc" = "124" ] && return 124
  analyze "$st" "$ANALYSIS" "$out"
  return 0
}

for st in $STAGES; do
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then log "plazo vencido; quedan sin correr: $st y siguientes"; break; fi
  run_stage "$st"; rc=$?
  if [ "$rc" = "124" ] || [ "$rc" = "99" ]; then log "corte por plazo en $st"; break; fi
done
final_reports
log "fin. Informes: $RES/f2b_final$SUFFIX/informe_Q.md (primario), $RES/f2b_final$SUFFIX/v1/informe_Q.md (evaluador v1) y $RES/f2b_final$SUFFIX/v2/informe_v2.md"

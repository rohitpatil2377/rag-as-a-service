{{/* Common name + label helpers for the Federated RAG umbrella chart. */}}

{{- define "rag.labels" -}}
app.kubernetes.io/part-of: ragaas
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
rag.local/department: {{ .Values.department | quote }}
{{- end -}}

{{- define "rag.apiImage" -}}
{{- if .Values.api.image -}}{{ .Values.api.image }}{{- else -}}{{ .Values.global.imageRegistry }}/api:{{ .Values.global.imageTag }}{{- end -}}
{{- end -}}

{{- define "rag.consoleImage" -}}
{{- if .Values.console.image -}}{{ .Values.console.image }}{{- else -}}{{ .Values.global.imageRegistry }}/console:{{ .Values.global.imageTag }}{{- end -}}
{{- end -}}

{{/*
  ONE switch for the whole-release service-mesh: global.linkerd.enabled. It lives under
  `global` so a SINGLE value meshes the parent workloads (api, console) AND propagates to the
  sovereign-node subchart automatically (Helm merges `global` into every subchart's .Values).
  Returns "true"/"false"; call as: {{- if eq (include "rag.linkerdEnabled" .) "true" }}.
  dig + default dict keeps it nil-safe if `global` is ever unset.
*/}}
{{- define "rag.linkerdEnabled" -}}
{{- dig "linkerd" "enabled" false (.Values.global | default dict) -}}
{{- end -}}

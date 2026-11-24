{{/*
Common naming + label helpers for the sovereign-node chart. One release per department,
so the release name is expected to be the department (e.g. `helm install hr ...`).
*/}}

{{- define "sovereign-node.name" -}}
sovereign-node
{{- end -}}

{{- define "sovereign-node.fullname" -}}
{{- printf "%s-sovereign-node" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Selector labels — the STABLE identity used by Service, Deployment, and (critically) the
NetworkPolicy peer/self selectors. The `name` label is shared across every department's
release so mesh members recognise each other; `department` distinguishes the pod.
*/}}
{{- define "sovereign-node.selectorLabels" -}}
app.kubernetes.io/name: {{ include "sovereign-node.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "sovereign-node.labels" -}}
{{ include "sovereign-node.selectorLabels" . }}
app.kubernetes.io/part-of: sovereign-federation
app.kubernetes.io/managed-by: {{ .Release.Service }}
rag.local/department: {{ required "department is required (set .Values.department)" .Values.department | quote }}
{{- end -}}

{{- define "sovereign-node.identitySecretName" -}}
{{- if .Values.identity.secretName -}}
{{ .Values.identity.secretName }}
{{- else -}}
{{ include "sovereign-node.fullname" . }}-identity
{{- end -}}
{{- end -}}

import type { ModelOption } from "../../hooks/useModels";

interface ModelSelectorProps {
    value: string
    onChange: (modelValue: string) => void
    models: ModelOption[]
}

function ModelSelector({ value, onChange, models }: ModelSelectorProps) {
    return (
        <select 
            id="modelSelector" 
            value={value}
            onChange={(e) => onChange(e.target.value)}
        >
            {models.map(model => (
                <option key={model.apiValue} value={model.apiValue}>
                    {model.name}
                </option>
            ))}
        </select>
    )
}

export default ModelSelector
import { AVAILABLE_MODELS } from '../../constants/models'

interface ModelSelectorProps {
    value: string
    onChange: (modelValue: string) => void
}

function ModelSelector({ value, onChange }: ModelSelectorProps) {
    return (
        <select 
            id="modelSelector" 
            value={value}
            onChange={(e) => onChange(e.target.value)}
        >
            {AVAILABLE_MODELS.map(model => (
                <option key={model.apiValue} value={model.apiValue}>
                    {model.name}
                </option>
            ))}
        </select>
    )
}

export default ModelSelector
import { useModels } from "../../hooks/useModels";

interface ModelSelectorProps {
    value: string
    onChange: (modelValue: string) => void
}

function ModelSelector({ value, onChange }: ModelSelectorProps) {
    const { models } = useModels();

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
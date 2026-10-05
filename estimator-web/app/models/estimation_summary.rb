class EstimationSummary
  include ActiveModel::Model
  include ActiveModel::Attributes

  attribute :total_hours, :integer
  attribute :total_cost_usd, :integer
  attribute :team_size, :string
  attribute :estimated_duration_weeks, :integer
  attribute :hourly_rate, :integer
  attribute :assumptions, default: -> { [] }

  def assumptions
    super || []
  end
end
